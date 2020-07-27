# %%
import math
import torch
import torch.nn as nn
import pickle
from loguru import logger

from dataset import dataloader, vocab_size, token_index
from tqdm import tqdm

from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = build_model(device, vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
ntokens = vocab_size

logger.info("Built model")
logger.info("Loading embedding weights...")
embeddings_dict = torch.load(open("embeddings.pickle", "rb"))
embeddings_weight = torch.zeros((ntokens, 100))
for i, word in enumerate(tqdm(token_index.index_to_token)):
    try:
        embeddings_weight[i] = embeddings_dict[word]
    except KeyError:
        embeddings_weight[i] = torch.rand(size=(100,))

logger.info("Prepared weights...")

model.init_weights(embeddings_weight)


# %%
def train():
    model.train()
    logger.info("Loaded weights into encoder.")

    logger.info("Training")
    total_loss = 0.0
    for batch, data in enumerate(tqdm(dataloader)):
        x, y, mask = data
        # mask = mask.reshape((mask.shape[1], mask.shape[0]))
        optimizer.zero_grad()
        output = model(x, attention_mask=mask)
        loss_x = output.view(-1, ntokens)
        loss_y = y.view(-1)
        loss = loss_fn(loss_x, loss_y)
        loss.backward()
        total_loss += loss.item()
        nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        if batch > 0 and batch % 100 == 0:
            logger.info(f"Batch {batch}: train loss {total_loss / 100}")
            total_loss = 0


# %%
# def evaluate(eval_model, data_source):
#     eval_model.eval()
#     total_loss = 0
#     ntokens = vocab_size
#     with torch.no_grad():
#         for i in range(0, data_source.size(0) - 1, bptt):
#             data, targets = get_batch(data_source, i)
#             output = eval_model(data)
#             output_flat = output.view(-1, ntokens)
#             total_loss += len(data) * loss_fn(output_flat, targets).item()
#     return total_loss / (len(data_source) - 1)


num_epochs = 3

for epoch in range(1, num_epochs + 1):
    train()
    # val_loss = evaluate(model, val_data)
    val_loss = None
    # scheduler.step()

    logger.info(f"Epoch {epoch}: Val Loss: {val_loss}")

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": val_loss,
        },
        "saved_models/model.pt",
    )
