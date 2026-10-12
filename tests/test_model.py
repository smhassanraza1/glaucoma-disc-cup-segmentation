import torch

from glaucoma.losses import segmentation_loss, soft_dice_loss
from glaucoma.model import UNet


def test_unet_output_has_requested_channels_at_input_resolution():
    for n_classes in (1, 2):
        model = UNet(pretrained=False, n_classes=n_classes).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64, 96))
        assert out.shape == (2, n_classes, 64, 96)


def test_soft_dice_loss_is_zero_for_perfect_and_high_for_wrong():
    target = torch.zeros(1, 2, 16, 16)
    target[:, :, 4:12, 4:12] = 1
    perfect = (target * 40) - 20  # confident logits matching the target
    assert soft_dice_loss(perfect, target) < 0.01
    assert soft_dice_loss(-perfect, target) > 0.9


def test_unet_can_overfit_one_synthetic_image():
    torch.manual_seed(0)
    model = UNet(pretrained=False)
    image = torch.randn(1, 3, 64, 64)
    target = torch.zeros(1, 2, 64, 64)
    target[:, 0, 16:48, 16:48] = 1  # disc
    target[:, 1, 26:38, 26:38] = 1  # cup, inside the disc
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = segmentation_loss(model(image), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
