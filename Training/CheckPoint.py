import torch
import torch
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,  file_path: str) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(src, model, optimizer):
    check_point = torch.load(src)
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    return check_point['epoch']