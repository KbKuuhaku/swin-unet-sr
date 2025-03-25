from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import click
import cv2
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

N_THREADS = 50


def resize_img(img_file: Path, tar_res: int) -> np.ndarray:
    img = cv2.imread(str(img_file))
    src_h = img.shape[0]

    resize_factor = tar_res / src_h

    resized_img = cv2.resize(
        img,
        (0, 0),
        fx=resize_factor,
        fy=resize_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    return resized_img


def get_res_img_dir(img_dir: Path, tar_res: int, res_type: str) -> Path:
    dataset_name = img_dir.stem.split("_")[0]

    return img_dir.with_stem(f"{dataset_name}_{res_type}_{tar_res}")


def create_progress() -> Progress:
    """
    Customizable progress bar from `rich.progress`
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def generating_resized_img(
    pbar: Progress,
    task_id: TaskID,
    img_file: Path,
    tar_img_dir: Path,
    target_res: int,
    res_type: str,
) -> None:
    low_res_img = resize_img(img_file, target_res)

    # Save image
    resized_img_file = tar_img_dir / f"{img_file.stem}.png"
    cv2.imwrite(str(resized_img_file), low_res_img)
    pbar.update(
        task_id,
        advance=1,
        description=f"[yellow][{task_id}] Generating {res_type} {resized_img_file}",
    )


@click.command()
@click.argument("img_dir")
@click.option("--low-res", default=128, show_default=True, help="Low resolution")
@click.option("--high-res", default=512, show_default=True, help="High resolution")
def main(img_dir: str, low_res: int, high_res: int) -> None:
    src_img_dir = Path(img_dir)
    low_res_img_dir = get_res_img_dir(src_img_dir, low_res, "lr")
    high_res_img_dir = get_res_img_dir(src_img_dir, high_res, "hr")

    low_res_img_dir.mkdir(parents=True, exist_ok=True)
    high_res_img_dir.mkdir(parents=True, exist_ok=True)

    click.echo("#" * 100)
    click.echo(f"Directory of low-resoultion images: {low_res_img_dir}")
    click.echo(f"Directory of high-resoultion images: {high_res_img_dir}")
    click.echo("#" * 100)

    glob_pattern = "*.[jpJP][npNP][egEG]"
    total_src_files = len(list(src_img_dir.glob(glob_pattern)))

    def submit_imgs(
        executor: Executor,
        pbar: Progress,
        low_res_img_dir: Path,
        low_res: int,
        res_type="Low Resolution",
    ) -> None:
        task_id = pbar.add_task("", total=total_src_files)
        for img_file in src_img_dir.glob(glob_pattern):
            executor.submit(
                generating_resized_img,
                pbar,
                task_id,
                img_file,
                low_res_img_dir,
                low_res,
                res_type,
            )

    with create_progress() as pbar:
        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            submit_imgs(
                executor,
                pbar,
                low_res_img_dir,
                low_res,
                res_type="Low Resolution",
            )
            submit_imgs(
                executor,
                pbar,
                high_res_img_dir,
                high_res,
                res_type="High Resolution",
            )

    n_low_res = len(list(low_res_img_dir.glob(glob_pattern)))
    n_high_res = len(list(high_res_img_dir.glob(glob_pattern)))
    click.echo(f"# Low-resolution images: {n_low_res}")
    click.echo(f"# High-resolution images: {n_high_res}")


if __name__ == "__main__":
    main()
