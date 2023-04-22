# import autotokenizer
import json
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import T5Tokenizer, AutoModel

class MathWordProblemDataset(Dataset):
    def __init__(self, problems, tokenizer, device):
        self.problems = problems
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        mwp = problem['prob']
        equations = problem['equations']
        local_gt = problem['gt']

        # Add prefixes for MWP and equations
        mwp = 'pre ' + mwp
        equations = ['suffi ' + equation for equation in equations]

        mwp_encoded = self.tokenizer(mwp, return_tensors="pt", padding="max_length", max_length=200).to(self.device)
        equations_encoded = [self.tokenizer(equation, return_tensors="pt", padding="max_length", max_length=64).to(self.device) for equation in equations]
        return mwp_encoded, equations_encoded, torch.tensor(local_gt, dtype=torch.float)

def contrastive_loss(problems_embeddings, equations_embeddings, gt, temperature=1.0):
    problems_embeddings = problems_embeddings.unsqueeze(1)
    scores = torch.matmul(problems_embeddings, equations_embeddings.transpose(1, 2)) / temperature
    scores = scores.squeeze(1)
    logits = torch.sigmoid(scores)
    # Compute the loss using binary cross-entropy
    loss = torch.nn.BCELoss()(logits, gt)        
    return loss

def load_json(path):
    with open(path) as f:
        return json.load(f)

def squeeze_batch(batch):
    for k, v in batch.items():
        batch[k] = v.squeeze(1)
    return batch

def main():
    # create python arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="kalpeshk2011/rankgen-t5-xl-all")
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--trainfile", type=str, default="traint5_preds.json")
    parser.add_argument("--valfile", type=str, default="t5_preds.json")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    model_path = args.model_path
    model_size = args.model_size
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    trainfile = args.trainfile
    valfile = args.valfile

    assert model_path in ["kalpeshk2011/rankgen-t5-xl-all", "kalpeshk2011/rankgen-t5-xl-pg19", "kalpeshk2011/rankgen-t5-base-all", "kalpeshk2011/rankgen-t5-large-all"]
    if model_size is None:
        if "t5-large" in model_path or "t5_large" in model_path:
            model_size = "large"
        elif "t5-xl" in model_path or "t5_xl" in model_path:
            model_size = "xl"
        else:
            model_size = "base"
    else:
        model_size = model_size

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.train()
    tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{model_size}")

    train_problems = load_json(trainfile)  # Your training data
    val_problems = load_json(valfile)  # Your validation data

    train_dataset = MathWordProblemDataset(train_problems, tokenizer, device)
    val_dataset = MathWordProblemDataset(val_problems, tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)


    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for mwp_input, equations_input, gt in tqdm(train_loader):
            optimizer.zero_grad()

            # Encode the MWP
            mwp_input = squeeze_batch(mwp_input)
            mwp_encoded = model(**mwp_input)
            # Encode the equations
            equations_input = [squeeze_batch(equation) for equation in equations_input]
            equation_embeddings = [model(**equation) for equation in equations_input]
            equations_encoded = torch.stack(equation_embeddings, dim=1)

            # Compute the contrastive loss
            loss = contrastive_loss(mwp_encoded, equations_encoded, gt)
            print(f"loss is {loss}")

            # Backpropagate and update the weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        # Evaluate the model on the validation set
        model.eval()
        total_val_loss = 0.0
        with torch.inference_mode():
            for mwp_input, equations_input, gt in val_loader:
                mwp_input = squeeze_batch(mwp_input)
                mwp_encoded = model(**mwp_input)
                equations_input = [squeeze_batch(equation) for equation in equations_input]
                equation_embeddings = [model(**equation) for equation in equations_input]
                equations_encoded = torch.stack(equation_embeddings, dim=1)

                # Compute the contrastive loss for the validation set
                val_loss = contrastive_loss(mwp_encoded, equations_encoded, gt)
                total_val_loss += val_loss.item()

        print(f"Validation Epoch {epoch+1}, Loss: {total_val_loss / len(val_loader)}")

if __name__ == "__main__":
    main()