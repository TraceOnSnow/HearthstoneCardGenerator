import argparse
import json
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageFilter, ImageOps

CARD_TYPE_TO_MASK_NAME = {
    4: "minion",  # Minion
    5: "spell",   # Spell
}


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


def _build_binary_mask(mask_trim: Image.Image, alpha_threshold: int, invert_mask: bool, erode_pixels: int = 0) -> Image.Image:
    mask_alpha = mask_trim.getchannel("A")
    binary_mask = mask_alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
    if invert_mask:
        binary_mask = ImageOps.invert(binary_mask)
    if erode_pixels > 0:
        for _ in range(erode_pixels):
            binary_mask = binary_mask.filter(ImageFilter.MinFilter(3))
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
    invert_mask: bool = True,
    auto_fix_polarity: bool = False,
    erode_pixels: int = 8,
) -> Image.Image:
    """Extract card art by aligning card body and mask body, then apply mask window."""
    card_resized, mask_trim = prepare_aligned_images(
        card_img=card_img,
        mask_img=mask_img,
        alpha_threshold=alpha_threshold,
    )

    target_w, target_h = mask_trim.size

    binary_mask = _build_binary_mask(mask_trim, alpha_threshold, invert_mask, erode_pixels)
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
    invert_mask: bool = True,
    auto_fix_polarity: bool = False,
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
    print(f"Saved: {output_path}")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    minion_mask_path: Path,
    spell_mask_path: Path,
    alpha_threshold: int = 1,
    add_padding: int = 24,
    keep_transparency: bool = True,
    invert_mask: bool = True,
    auto_fix_polarity: bool = False,
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
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Failed: {card_file.name} -> {e}")


def process_bulk(
    jsonl_path: Path,
    card_images_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    alpha_threshold: int = 1,
    add_padding: int = 24,
    keep_transparency: bool = True,
    invert_mask: bool = True,
    auto_fix_polarity: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    masks: dict[int, Image.Image] = {}
    for type_id, mask_name in CARD_TYPE_TO_MASK_NAME.items():
        mask_path = mask_dir / f"{mask_name}_mask.png"
        if mask_path.exists():
            masks[type_id] = Image.open(mask_path).convert("RGBA")
        else:
            print(f"Warning: mask not found for {mask_name} ({mask_path}), skipping type {type_id}")

    cards: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            card_id = card.get("id")
            card_type = card.get("cardTypeId")
            if isinstance(card_id, int) and card_type in masks:
                cards.append(card)

    total = len(cards)
    done = 0
    skipped = 0
    failed = 0

    for card in cards:
        card_id = card["id"]
        card_type = card["cardTypeId"]
        card_path = card_images_dir / f"{card_id}.png"
        out_path = output_dir / f"{card_id}.png"

        if out_path.exists():
            skipped += 1
            continue

        if not card_path.exists():
            skipped += 1
            continue

        try:
            card_img = Image.open(card_path).convert("RGBA")
            art = extract_art_with_mask(
                card_img=card_img,
                mask_img=masks[card_type],
                alpha_threshold=alpha_threshold,
                add_padding=add_padding,
                keep_transparency=keep_transparency,
                invert_mask=invert_mask,
                auto_fix_polarity=auto_fix_polarity,
            )
            art.save(out_path)
            done += 1
            if done % 100 == 0:
                print(f"[{done + skipped + failed}/{total}] Processed {done} cards...")
        except Exception as e:
            failed += 1
            print(f"Failed: {card_id} -> {e}")

    print(f"Done. Processed: {done}, Skipped: {skipped}, Failed: {failed}, Total: {total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop Hearthstone art region by transparent mask.")

    parser.add_argument("--mode", choices=["one", "folder", "bulk"], default="folder")

    parser.add_argument("--card", type=Path, help="Single card PNG path (mode=one).")
    parser.add_argument("--mask", type=Path, help="Single mask PNG path (mode=one).")
    parser.add_argument("--out", type=Path, help="Output path (mode=one).")

    parser.add_argument("--input-dir", type=Path, default=Path("data/sample_img/cards"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cropped_cards"))
    parser.add_argument("--minion-mask", type=Path, default=Path("data/sample_img/masks/minion_mask.png"))
    parser.add_argument("--spell-mask", type=Path, default=Path("data/sample_img/masks/spell_mask.png"))

    parser.add_argument("--jsonl", type=Path, default=Path("data/cards_collectible.jsonl"),
                        help="JSONL file with card data (mode=bulk).")
    parser.add_argument("--card-images-dir", type=Path, default=Path("data/card_images"),
                        help="Directory with downloaded card images (mode=bulk).")
    parser.add_argument("--mask-dir", type=Path, default=Path("data/sample_img/masks"),
                        help="Directory with mask templates (mode=bulk).")

    parser.add_argument("--alpha-threshold", type=int, default=1)
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--solid-bg", action="store_true", help="Output RGB with black background.")
    parser.add_argument("--no-invert-mask", action="store_true", help="Disable default mask inversion.")
    parser.add_argument("--auto-fix", action="store_true", help="Enable automatic mask polarity fix.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_transparency = not args.solid_bg
    invert_mask = not args.no_invert_mask
    auto_fix_polarity = args.auto_fix

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
            invert_mask=invert_mask,
            auto_fix_polarity=auto_fix_polarity,
        )
    elif args.mode == "bulk":
        process_bulk(
            jsonl_path=args.jsonl,
            card_images_dir=args.card_images_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            alpha_threshold=args.alpha_threshold,
            add_padding=args.padding,
            keep_transparency=keep_transparency,
            invert_mask=invert_mask,
            auto_fix_polarity=auto_fix_polarity,
        )
    else:
        process_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            minion_mask_path=args.minion_mask,
            spell_mask_path=args.spell_mask,
            alpha_threshold=args.alpha_threshold,
            add_padding=args.padding,
            keep_transparency=keep_transparency,
            invert_mask=invert_mask,
            auto_fix_polarity=auto_fix_polarity,
        )


if __name__ == "__main__":
    main()