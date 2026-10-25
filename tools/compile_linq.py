"""Подготовка YOLO к LinQ TPU: разрез модели, калибровка на реальных кадрах, компиляция в программу .tpu.

Почему разрез. Компилятор LinQ поддерживает YOLO не целиком: в документации SDK (раздел
Supported models) у yolov8n_pose, yolo12m и др. сноска — модель «обрезана» перед концом головы,
а остаток выполняется в постобработке. Мы режем так же: после последних свёрток головы.

    models/ppe.onnx  ──split──►  ppe_head.onnx  (вся сеть из свёрток)  ──SDK──►  ppe_head.tpu   (на TPU)
                                 ppe_tail.onnx  (reshape, DFL, декодирование боксов)          (на CPU, onnxruntime)

Шаги:
    python tools/compile_linq.py split   --onnx models/ppe.onnx            # любой ОС: разрез + проверка
    python tools/compile_linq.py compile --onnx models/ppe.onnx --calib-dir datasets/ppe/val/images
                                                                           # Linux + Python 3.12 + SDK LinQ

SDK (Framework.zip -> tpu_framework/packets) ставится только на Linux x86_64 с Python 3.12:
    python3.12 -m venv venv && source venv/bin/activate
    pip install -U pip && pip install -r tpu_framework/packets/requirements.txt   (из папки packets)
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.backends import YoloRawBackend, read_onnx_names  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------- разрез (работает на любой ОС) ----------

def find_head_cuts(model) -> list[str]:
    """Выходы последних свёрток головы YOLOv8/YOLO11: Conv последнего модуля (Detect), результат
    которой сразу идёт в Reshape. Свёртку DFL (/dfl/) пропускаем — она часть декодирования и уходит в хвост.
    Только последний модуль: в backbone YOLO11 (блок внимания C2PSA) тоже есть Conv -> Reshape."""
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    head_prefix = "/".join(model.graph.node[-1].name.split("/")[:2]) + "/"   # например "/model.23/"
    cuts = []
    for node in model.graph.node:
        if node.op_type != "Conv" or "/dfl/" in node.name or not node.name.startswith(head_prefix):
            continue
        out = node.output[0]
        if any(c.op_type == "Reshape" for c in consumers.get(out, [])):
            cuts.append(out)
    if not cuts:
        raise RuntimeError("Не нашёл свёртки головы YOLO — укажи --cut вручную")
    return cuts


def split(onnx_path: Path, cuts: list[str] | None = None) -> tuple[Path, Path, list[str]]:
    """Режет ONNX на голову (вход -> cuts) и хвост (cuts -> исходный выход)."""
    import onnx
    from onnx.utils import extract_model

    model = onnx.load(str(onnx_path))
    cuts = cuts or find_head_cuts(model)
    inp, out = model.graph.input[0].name, model.graph.output[0].name
    head, tail = onnx_path.with_name(onnx_path.stem + "_head.onnx"), onnx_path.with_name(onnx_path.stem + "_tail.onnx")
    extract_model(str(onnx_path), str(head), [inp], cuts)
    extract_model(str(onnx_path), str(tail), cuts, [out])

    # имена классов нужны бэкенду — переносим метаданные ultralytics в голову и хвост.
    # Версию формата (IR) оставляем исходной: SDK LinQ использует onnx~=1.16 и не читает слишком новые файлы.
    for p in (head, tail):
        part = onnx.load(str(p))
        del part.metadata_props[:]
        part.metadata_props.extend(model.metadata_props)
        part.ir_version = model.ir_version
        onnx.save(part, str(p))
    return head, tail, cuts


def verify_split(onnx_path: Path, head: Path, tail: Path, image: np.ndarray) -> float:
    """Голова + хвост должны давать ровно тот же выход, что и исходная модель. Возвращает макс. расхождение."""
    import onnxruntime as ort

    opts = ["CPUExecutionProvider"]
    full, h, t = (ort.InferenceSession(str(p), providers=opts) for p in (onnx_path, head, tail))
    size = full.get_inputs()[0].shape[2]
    blob = preprocess(image, size)
    ref = full.run(None, {full.get_inputs()[0].name: blob})[0]
    mid = dict(zip([o.name for o in h.get_outputs()], h.run(None, {h.get_inputs()[0].name: blob})))
    got = t.run(None, {i.name: mid[i.name] for i in t.get_inputs()})[0]
    return float(np.abs(ref - got).max())


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Та же подготовка кадра, что и при инференсе (YoloRawBackend.preprocess): калибровать на другой — нельзя."""
    b = YoloRawBackend({"imgsz": size, "conf": 0, "iou": 0}, {})
    return b.preprocess(image)[0]


def calibration_tensor(calib_dir: Path, size: int, n: int) -> np.ndarray:
    imgs = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    random.Random(0).shuffle(imgs)
    blobs = [preprocess(img, size) for img in (cv2.imread(str(p)) for p in imgs[:n]) if img is not None]
    if not blobs:
        raise RuntimeError(f"В {calib_dir} нет картинок для калибровки")
    return np.concatenate(blobs, axis=0).astype(np.float32)


# ---------- квантование и компиляция (Linux + SDK LinQ) ----------

def compile_tpu(head: Path, calib: np.ndarray, preset: str, out: Path, percentile: float | None):
    """По образцу tpu_framework/compile.py из SDK, но с калибровкой на реальных кадрах вместо шума.
    Возвращает (путь к .tpu, квантованная модель) — модель нужна для проверки точности INT8 на CPU."""
    import onnx
    from tpu_framework import O1, O5, TPU_H1M, Network, QuantizedModel, RegularModel, TpuProgram, compiler
    from tpu_framework import preliminary_model_conditioning

    work = out.parent / (head.stem + "_build")
    work.mkdir(parents=True, exist_ok=True)

    print("1/3. Оптимизация графа")
    model = onnx.load(str(head))
    # как в compile.py: убрать из входов то, что на самом деле веса (initializer)
    init = {i.name for i in model.graph.initializer}
    for inp in [i for i in model.graph.input if i.name in init]:
        model.graph.input.remove(inp)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}
    if any(0 in s for s in shapes.values()):
        raise RuntimeError(f"Динамическая форма входа {shapes} не поддерживается — экспортируй с dynamic=False")
    model = preliminary_model_conditioning(model)
    onnx.save(model, str(work / "opt.onnx"))

    print(f"2/3. Калибровка на {len(calib)} реальных кадрах и квантование в INT8")
    model = onnx.load(str(work / "opt.onnx"))
    input_name = model.graph.input[0].name
    regular = RegularModel(model=model)
    kwargs = {"samplewise_percentile": percentile} if percentile else {}
    thresholds = regular.calibrate_torch(calibration_data={input_name: calib}, **kwargs)
    RegularModel.dump_thresholds(str(work / "thresholds.json"), thresholds)
    quant = regular.quantize(thresholds)
    qm_name = head.stem + ".qm.onnx"
    quant.save(file_dir=str(work), file_name=qm_name)

    print(f"3/3. Компиляция под TPU_H1M, пресет {preset}")
    quantized = QuantizedModel.load(str(work / qm_name))
    network, _ = Network.from_quantized_model(quantized)
    executable, tensors = compiler.compile_(hardware_parameters=TPU_H1M, network=network,
                                            parameters={"o1": O1, "o5": O5}[preset])
    TpuProgram.from_executable(executable, tensors).to_file(str(out))
    print(f"Готово: {out} ({out.stat().st_size / 1e6:.1f} МБ)")
    return out, quantized


def check_int8_on_cpu(quantized, onnx_path: Path, head: Path, tail: Path, images: list[Path]) -> dict:
    """Точность после квантования — ещё до стенда. SDK умеет выполнить квантованную модель на CPU
    (QuantizedModel.as_torch_graph, раздел End-To-End Test документации) — это та же арифметика INT8,
    что и на TPU. Голова INT8 + хвост на CPU сравниваются с исходной моделью FP32 через tools/compare.match."""
    import onnx
    import onnxruntime as ort
    import torch
    from dnn_quant.api import TpuDtype
    from tpu_framework import TPU_H1M

    from app.backends import Detection, OnnxBackend
    from tools.compare import match

    g = quantized.as_torch_graph(batch_size=1, cache_word_length=TPU_H1M.tpu_parameters.cache_word_length)
    gm = g.gm.eval()
    anchors = g.get_outputs_anchors() | g.get_inputs_anchors()   # имя в ONNX -> имя в torch-графе
    head_model = onnx.load(str(head))
    in_name = head_model.graph.input[0].name
    out_names = [o.name for o in head_model.graph.output]

    tail_sess = ort.InferenceSession(str(tail), providers=["CPUExecutionProvider"])
    tail_shapes = {i.name: tuple(i.shape) for i in tail_sess.get_inputs()}
    ref = OnnxBackend({"imgsz": 640, "conf": 0.35, "iou": 0.5}, path=str(onnx_path))

    n_ref = n_q = n_match = 0
    for p in images:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        blob, r, pad = ref.preprocess(frame)
        x = quantized.rescale({anchors[in_name]: blob}, TpuDtype.Input)[anchors[in_name]]
        with torch.no_grad():
            outs = gm(torch.as_tensor(x))
        outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        outs = quantized.rescale({n: o.detach().cpu().numpy() for n, o in zip(out_names, outs)}, TpuDtype.Output)
        # выходы раскладываем по входам хвоста по форме — она у каждого из 6 тензоров своя
        by_shape = {tuple(v.shape): np.asarray(v, dtype=np.float32) for v in outs.values()}
        feeds = {name: by_shape[shape] for name, shape in tail_shapes.items()}
        dets_q = ref.postprocess(tail_sess.run(None, feeds)[0], r, pad, frame.shape)
        dets_ref = ref.detect(frame)
        n_ref, n_q, n_match = n_ref + len(dets_ref), n_q + len(dets_q), n_match + len(match(dets_ref, dets_q))

    res = {"recall": n_match / n_ref if n_ref else 1.0, "precision": n_match / n_q if n_q else 1.0,
           "ref": n_ref, "int8": n_q, "matched": n_match}
    print(f"INT8 на CPU против FP32: recall {res['recall']:.3f}, precision {res['precision']:.3f} "
          f"(детекций FP32 {n_ref}, INT8 {n_q}, совпало {n_match})")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["split", "compile"])
    ap.add_argument("--onnx", default=str(ROOT / "models" / "ppe.onnx"))
    ap.add_argument("--cut", nargs="*", help="имена тензоров разреза (по умолчанию — найти автоматически)")
    ap.add_argument("--image", default=str(ROOT / "data" / "sample.jpg"), help="картинка для проверки разреза")
    ap.add_argument("--calib-dir", help="кадры для калибровки (обычно val датасета или кадры с камер)")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--preset", choices=["o1", "o5"], default="o1",
                    help="o1 — надёжно; o5 — быстрее, но может не скомпилироваться (тогда вернуться к o1)")
    ap.add_argument("--percentile", type=float, default=None,
                    help="samplewise_percentile калибровки, напр. 99.9 — устойчивость к выбросам")
    ap.add_argument("--check", type=int, default=50,
                    help="на скольких кадрах сравнить INT8 (на CPU) с исходной моделью; 0 — не проверять")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    head, tail, cuts = split(onnx_path, args.cut)
    names = read_onnx_names(str(onnx_path))
    print(f"Разрез: {len(cuts)} тензоров, классов {len(names)}")
    for c in cuts:
        print("   ", c)
    image = cv2.imread(args.image)
    if image is not None:
        diff = verify_split(onnx_path, head, tail, image)
        print(f"Проверка: голова+хвост против исходной модели, макс. расхождение {diff:.2e}",
              "OK" if diff < 1e-4 else "ОШИБКА")
    print(f"Голова (на TPU): {head}\nХвост (на CPU): {tail}")

    if args.step == "compile":
        if not args.calib_dir:
            sys.exit("Для компиляции нужен --calib-dir с реальными кадрами")
        import onnx
        size = [d.dim_value for d in onnx.load(str(head)).graph.input[0].type.tensor_type.shape.dim][2]
        calib = calibration_tensor(Path(args.calib_dir), size, args.calib)
        _, quantized = compile_tpu(head, calib, args.preset, head.with_suffix(".tpu"), args.percentile)

        if args.check > 0:
            imgs = sorted(p for p in Path(args.calib_dir).iterdir() if p.suffix.lower() in IMG_EXT)
            random.Random(1).shuffle(imgs)
            try:
                check_int8_on_cpu(quantized, onnx_path, head, tail, imgs[:args.check])
            except Exception as e:   # проверка вспомогательная — .tpu уже скомпилирован
                print(f"Проверку INT8 на CPU выполнить не удалось ({type(e).__name__}: {e}). "
                      f"Точность проверим на стенде через tools/compare.py")


if __name__ == "__main__":
    main()
