from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FIGURES = {
    "three-tasks": "Fig_Three_Task_QA",
    "architecture": "architecture",
    "qualitative": "Fig_Qualitative_Analysis",
    "state-transition": "Fig_State_Transition",
    "benchmarks": "Fig_Benchmark_Overview",
    "data-full": "Fig_Data_Pipeline",
    "memory-examples": "Fig_Appendix_Memory_Perception",
    "proactive-examples": "Fig_Appendix_Proactive_Response",
}

def build_logo():
    original = ASSETS / "OneStreamer_Logo.png"
    if not original.exists():
        return
    logo = Image.open(original).convert("RGBA")
    logo.thumbnail((512, 512), Image.Resampling.LANCZOS)
    logo.save(ASSETS / "OneStreamer_Logo.webp", quality=94, method=6)
    logo.resize((64, 64), Image.Resampling.LANCZOS).save(ASSETS / "favicon.png")
    logo.resize((180, 180), Image.Resampling.LANCZOS).save(ASSETS / "apple-touch-icon.png")
    logo.save(ASSETS / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])


def main():
    parser = ArgumentParser()
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    for folder in ("figures", "frames", "fonts"):
        (ASSETS / folder).mkdir(parents=True, exist_ok=True)
    build_logo()
    with TemporaryDirectory(prefix="onestreamer-assets-") as temp:
        temporary = Path(temp)
        for output, source in FIGURES.items():
            prefix = temporary / output
            source_dir = ASSETS / "figures" if output == "architecture" else args.figures
            subprocess.run(["pdftoppm", "-singlefile", "-png", "-scale-to", "3000", str(source_dir / (source + ".pdf")), str(prefix)], check=True)
            im = Image.open(prefix.with_suffix(".png")).convert("RGB")
            im.save(ASSETS / "figures" / (output + ".webp"), quality=94, method=6)
            if output == "data-full":
                im.crop((0, 0, 982, im.height)).save(ASSETS / "figures" / "data-composition.webp", quality=95, method=6)
                im.crop((1003, 0, im.width, im.height)).save(ASSETS / "figures" / "data-pipeline.webp", quality=95, method=6)
        subprocess.run(["pdfimages", "-j", str(args.figures / "Fig_Qualitative_Analysis.pdf"), str(temporary / "frame")], check=True)
        for name, indices in {"memory": [0, 2, 4, 6, 8, 10, 12], "counting": [20, 22, 24, 26, 28, 30, 32]}.items():
            for step, source_index in enumerate(indices):
                im = Image.open(temporary / f"frame-{source_index:03}.jpg").convert("RGB")
                im = ImageOps.fit(im, (960, 540))
                im.save(ASSETS / "frames" / f"{name}-{step}.webp", quality=87, method=6)
                im.resize((240, 135)).save(ASSETS / "frames" / f"{name}-{step}-thumb.webp", quality=80, method=6)

if __name__ == "__main__":
    main()
