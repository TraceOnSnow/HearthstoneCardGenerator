import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps


def alpha_bbox(img: Image.Image, alpha_threshold: int = 1) -> Tuple[int, int, int, int]:
    """Return bbox of non-transparent area based on alpha channel."""
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    binary = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
    bbox = binary.getbbox()
    if bbox is None:
        raise ValueError("Image appears fully transparent.")
    return bbox


def trim_transparent_border(img: Image.Image, alpha_threshold: int = 1) -> Image.Image:
    """Crop image to the bbox of non-transparent pixels."""
    return img.crop(alpha_bbox(img, alpha_threshold=alpha_threshold))


def _build_binary_mask(mask_trim: Image.Image, alpha_threshold: int, invert_mask: bool) -> Image.Image:
    mask_alpha = mask_trim.getchannel("A")
    binary_mask = mask_alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
    if invert_mask:
        binary_mask = ImageOps.invert(binary_mask)
    return binary_mask


def _auto_fix_mask_polarity(binary_mask: Image.Image) -> Image.Image:
    """Ensure selected area is the center art region, not the outer frame."""
    if binary_mask.getbbox() is None:
        raise ValueError("Mask appears fully transparent after thresholding.")

    cx = binary_mask.width // 2
    cy = binary_mask.height // 2
    center_selected = binary_mask.getpixel((cx, cy)) > 0

    # Art window is expected near the center; if center is not selected,
    # mask polarity is likely reversed.
    if not center_selected:
        return ImageOps.invert(binary_mask)

    return binary_mask


def prepare_aligned_images(
    card_img: Image.Image,
    mask_img: Image.Image,
    alpha_threshold: int = 1,
) -> tuple[Image.Image, Image.Image]:
    """Trim transparent borders, then align card to mask canvas size."""
    card_trim = trim_transparent_border(card_img, alpha_threshold=alpha_threshold).convert("RGBA")
    mask_trim = trim_transparent_border(mask_img, alpha_threshold=alpha_threshold).convert("RGBA")

    target_w, target_h = mask_trim.size
    card_aligned = card_trim.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return card_aligned, mask_trim


def extract_art_with_mask(
    card_img: Image.Image,
    mask_img: Image.Image,
    alpha_threshold: int = 1,
    add_padding: int = 24,
    keep_transparency: bool = True,
    invert_mask: bool = False,
    auto_fix_polarity: bool = True,
) -> Image.Image:
    """Extract card art by aligning card body and mask body, then apply mask window."""
    card_resized, mask_trim = prepare_aligned_images(
        card_img=card_img,
        mask_img=mask_img,
        alpha_threshold=alpha_threshold,
    )

    target_w, target_h = mask_trim.size

    binary_mask = _build_binary_mask(mask_trim, alpha_threshold, invert_mask)
    if auto_fix_polarity:
        binary_mask = _auto_fix_mask_polarity(binary_mask)

    result = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    result.paste(card_resized, (0, 0), binary_mask)

    art_bbox = binary_mask.getbbox()
    if art_bbox is None:
        raise ValueError("Mask appears fully transparent after processing.")

    left, top, right, bottom = art_bbox
    left = max(0, left - add_padding)
    top = max(0, top - add_padding)
    right = min(target_w, right + add_padding)
    bottom = min(target_h, bottom + add_padding)

    cropped = result.crop((left, top, right, bottom))

    if keep_transparency:
        return cropped

    bg = Image.new("RGB", cropped.size, (0, 0, 0))
    bg.paste(cropped, mask=cropped.getchannel("A"))
    return bg


def process_one(
    card_path: Path,
    mask_path: Path,
    output_path: Path,
    alpha_threshold: int = 1,
    add_padding: int = 24,
    keep_transparency: bool = True,
    invert_mask: bool = False,
    auto_fix_polarity: bool = True,
) -> None:
    card_img = Image.open(card_path).convert("RGBA")
    mask_img = Image.open(mask_path).convert("RGBA")

    art = extract_art_with_mask(
        card_img=card_img,
        mask_img=mask_img,
        alpha_threshold=alpha_threshold,
        add_padding=add_padding,
        keep_transparency=keep_transparency,
        invert_mask=invert_mask,
        auto_fix_polarity=auto_fix_polarity,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    art.save(output_path)

    aligned_card, aligned_mask = prepare_aligned_images(
        card_img=card_img,
        mask_img=mask_img,
        alpha_threshold=alpha_threshold,
    )
    aligned_card_path = output_path.with_name(f"{output_path.stem}_aligned_card.png")
    aligned_mask_path = output_path.with_name(f"{output_path.stem}_aligned_mask.png")
    aligned_card.save(aligned_card_path)
    aligned_mask.save(aligned_mask_path)

    print(f"Saved: {output_path}")
    print(f"Saved: {aligned_card_path}")
    print(f"Saved: {aligned_mask_path}")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    minion_mask_path: Path,
    spell_mask_path: Path,
    alpha_threshold: int = 1,
    add_padding: int = 24,
    keep_transparency: bool = True,
    invert_mask: bool = False,
    auto_fix_polarity: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    minion_mask = Image.open(minion_mask_path).convert("RGBA")
    spell_mask = Image.open(spell_mask_path).convert("RGBA")

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    for card_file in png_files:
        name = card_file.stem.lower()
        if "minion" in name:
            mask_img = minion_mask
        elif "spell" in name:
            mask_img = spell_mask
        else:
            print(f"Skip {card_file.name}: cannot infer card type (need minion/spell in name).")
            continue

        try:
            card_img = Image.open(card_file).convert("RGBA")
            art = extract_art_with_mask(
                card_img=card_img,
                mask_img=mask_img,
                alpha_threshold=alpha_threshold,
                add_padding=add_padding,
                keep_transparency=keep_transparency,
                invert_mask=invert_mask,
                auto_fix_polarity=auto_fix_polarity,
            )
            save_path = output_dir / f"{card_file.stem}_art.png"
            art.save(save_path)

            aligned_card, aligned_mask = prepare_aligned_images(
                card_img=card_img,
                mask_img=mask_img,
                alpha_threshold=alpha_threshold,
            )
            aligned_card_path = output_dir / f"{card_file.stem}_aligned_card.png"
            aligned_mask_path = output_dir / f"{card_file.stem}_aligned_mask.png"
            aligned_card.save(aligned_card_path)
            aligned_mask.save(aligned_mask_path)

            print(f"Saved: {save_path}")
            print(f"Saved: {aligned_card_path}")
            print(f"Saved: {aligned_mask_path}")
        except Exception as e:
            print(f"Failed: {card_file.name} -> {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop Hearthstone art region by transparent mask.")

    parser.add_argument("--mode", choices=["one", "folder"], default="folder")

    parser.add_argument("--card", type=Path, help="Single card PNG path (mode=one).")
    parser.add_argument("--mask", type=Path, help="Single mask PNG path (mode=one).")
    parser.add_argument("--out", type=Path, help="Output path (mode=one).")

    parser.add_argument("--input-dir", type=Path, default=Path("cards"))
    parser.add_argument("--output-dir", type=Path, default=Path("cropped_cards"))
    parser.add_argument("--minion-mask", type=Path, default=Path("minion_mask.png"))
    parser.add_argument("--spell-mask", type=Path, default=Path("spell_mask.png"))

    parser.add_argument("--alpha-threshold", type=int, default=1)
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--solid-bg", action="store_true", help="Output RGB with black background.")
    parser.add_argument("--invert-mask", action="store_true", help="Force invert mask polarity.")
    parser.add_argument("--no-auto-fix", action="store_true", help="Disable automatic mask polarity fix.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_transparency = not args.solid_bg
    auto_fix_polarity = not args.no_auto_fix

    if args.mode == "one":
        if not args.card or not args.mask or not args.out:
            raise ValueError("mode=one requires --card --mask --out")
        process_one(
            card_path=args.card,
            mask_path=args.mask,
            output_path=args.out,
            alpha_threshold=args.alpha_threshold,
            add_padding=args.padding,
            keep_transparency=keep_transparency,
            invert_mask=args.invert_mask,
            auto_fix_polarity=auto_fix_polarity,
        )
        return

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        minion_mask_path=args.minion_mask,
        spell_mask_path=args.spell_mask,
        alpha_threshold=args.alpha_threshold,
        add_padding=args.padding,
        keep_transparency=keep_transparency,
        invert_mask=args.invert_mask,
        auto_fix_polarity=auto_fix_polarity,
    )


if __name__ == "__main__":
    main()