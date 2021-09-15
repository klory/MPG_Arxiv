import argparse
def get_args():
    parser = argparse.ArgumentParser(description='retrieval model parameters')
    parser.add_argument('--seed', default=8, type=int)
    parser.add_argument('--wandb', default=1, type=int, choices=[0,1])
    parser.add_argument('--dataset', default='pizza10', type=str, choices=['pizza10'])
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--num_hidden_layers', default=2, type=int)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--retrieved_type', default='image', choices=['recipe', 'image'])
    parser.add_argument('--retrieved_range', default=1000, type=int)
    parser.add_argument('--ckpt_path', default='')
    args = parser.parse_args()
    return args