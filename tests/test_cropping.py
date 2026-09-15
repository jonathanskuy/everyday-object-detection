"""Tests for crop_detection.

Every expected value here can be worked out by hand from the box coordinates.
Images are created in memory: a grey-free background (black) with a coloured
rectangle, so it's easy to check which pixels ended up in the crop.
"""

from PIL import Image

from object_detection.identification.cropping import PAD_COLOUR, crop_detection

RED = (255, 0, 0)
BLACK = (0, 0, 0)


def image_with_rectangle(size=(200, 100), box=(50, 20, 90, 80), colour=RED):
    """A black image with a filled rectangle covering `box` (x1, y1, x2, y2)."""
    image = Image.new("RGB", size, BLACK)
    image.paste(colour, box)
    return image


def test_tall_box_is_padded_to_a_square_with_the_object_centred():
    image = image_with_rectangle()  # red rectangle 40 wide, 60 tall
    crop = crop_detection(image, (50, 20, 90, 80), padding=0, min_size=1)

    assert crop.size == (60, 60)
    # 20 px of grey split evenly: 10 on the left, 10 on the right.
    assert crop.getpixel((0, 30)) == PAD_COLOUR
    assert crop.getpixel((9, 30)) == PAD_COLOUR
    assert crop.getpixel((10, 30)) == RED
    assert crop.getpixel((49, 30)) == RED
    assert crop.getpixel((50, 30)) == PAD_COLOUR


def test_wide_box_is_padded_above_and_below():
    image = image_with_rectangle(box=(20, 40, 120, 60))  # 100 wide, 20 tall
    crop = crop_detection(image, (20, 40, 120, 60), padding=0, min_size=1)

    assert crop.size == (100, 100)
    assert crop.getpixel((50, 39)) == PAD_COLOUR
    assert crop.getpixel((50, 40)) == RED
    assert crop.getpixel((50, 59)) == RED
    assert crop.getpixel((50, 60)) == PAD_COLOUR


def test_padding_adds_context_proportional_to_the_box():
    image = image_with_rectangle()
    # 10% of a 40 x 60 box: 4 px left and right, 6 px top and bottom -> 48 x 72.
    crop = crop_detection(image, (50, 20, 90, 80), padding=0.1, min_size=1)

    assert crop.size == (72, 72)
    # The added context is the image's black background, not grey padding.
    assert crop.getpixel((12, 36)) == BLACK   # 12 = (72 - 48) / 2 grey, then context starts
    assert crop.getpixel((16, 36)) == RED     # 4 px of context later, the object starts


def test_padding_is_clamped_at_the_image_edges():
    image = image_with_rectangle(size=(200, 100), box=(0, 0, 40, 30))
    crop = crop_detection(image, (0, 0, 40, 30), padding=0.5, min_size=1)

    # Left and top padding fall outside the image and are cut off:
    # region is x 0..60, y 0..45 -> 60 x 45, padded to 60 x 60.
    assert crop.size == (60, 60)


def test_fractional_coordinates_round_outward():
    image = image_with_rectangle()
    # x 50.4..89.6 must keep the partly covered pixels: 50..90 (40 px), y 20.2..79.9 -> 20..80.
    crop = crop_detection(image, (50.4, 20.2, 89.6, 79.9), padding=0, min_size=1)
    assert crop.size == (60, 60)


def test_boxes_smaller_than_min_size_are_rejected():
    image = image_with_rectangle()
    assert crop_detection(image, (50, 20, 54, 80), padding=0, min_size=5) is None   # 4 px wide
    assert crop_detection(image, (50, 20, 55, 80), padding=0, min_size=5) is not None


def test_box_outside_the_image_is_rejected():
    image = image_with_rectangle(size=(200, 100))
    assert crop_detection(image, (250, 20, 300, 80), padding=0, min_size=1) is None


def test_output_is_rgb_whatever_the_input_mode():
    for mode in ("L", "RGBA"):
        image = Image.new(mode, (200, 100))
        assert crop_detection(image, (10, 10, 50, 50), padding=0, min_size=1).mode == "RGB"


def test_input_image_is_not_modified():
    image = image_with_rectangle()
    before = image.tobytes()
    crop_detection(image, (50, 20, 90, 80), padding=0.2, min_size=1)
    assert image.tobytes() == before
