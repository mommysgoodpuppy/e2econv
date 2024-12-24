import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##############################################################################
#                          HYPERPARAMETERS
##############################################################################
vocab_size = 256
embed_size = 16
num_layers = 2  # number of residual conv blocks
channels = 64
kernel_size = 3
window_size = 4  # how many hidden states to keep
feedback_dim = 32  # each hidden state is condensed to this dim
lr = 0.0001
num_epochs = 150
grad_clip = 1.0

##############################################################################
#                             DATA
##############################################################################
text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean sed faucibus leo, non hendrerit lorem. Maecenas diam nisi, commodo nec tristique a, cursus vitae libero. Maecenas mollis, massa vel faucibus semper, metus nisl vehicula nisi, at vulputate mi erat ac lacus. Integer eget tempus velit. Donec efficitur accumsan egestas. Aliquam vel justo orci. Etiam augue quam, tincidunt eget ex vitae, varius eleifend turpis. Sed consequat diam elit, quis viverra odio congue id. Donec vehicula porta magna, tristique dictum ex. Morbi aliquam nulla eget tincidunt vestibulum. In dictum elit id neque scelerisque, eu dignissim ipsum laoreet. Duis eros lacus, aliquet vel interdum eu, pulvinar sit amet sapien. Aliquam erat volutpat. Duis facilisis erat eu odio viverra scelerisque. Vivamus pellentesque lacinia nunc. Curabitur tortor velit, convallis laoreet viverra sit amet, dapibus eu sem.
    """
encoded = [ord(c) for c in text]
# shape: (1, seq_len)
seq_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
B, seq_len = seq_tensor.shape


##############################################################################
#                         CAUSAL DILATED CONV BLOCK
##############################################################################
class CausalConv1d(nn.Conv1d):
    """1D conv with left-padding to ensure no future context."""

    def __init__(self, in_c, out_c, ksz, dilation=1, **kw):
        super().__init__(in_c, out_c, ksz, dilation=dilation, **kw)
        self.causal_pad = (ksz - 1) * dilation

    def forward(self, x):
        # x: [B, channels, T]
        x = F.pad(x, (self.causal_pad, 0))
        return super().forward(x)


class ResidualBlock(nn.Module):
    """One causal conv layer with dilation, plus residual + LayerNorm."""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, : x.size(2)]  # remove extra padding
        out = out.permute(0, 2, 1)  # [B, T, C]
        out = self.norm(out).permute(0, 2, 1)
        return x + self.act(out)  # residual


##############################################################################
#              WAVE-NET STYLE MODEL + ABSTRACT FEEDBACK WINDOW
##############################################################################
class WaveFeedbackModel(nn.Module):
    """
    - Embedding => stacked dilated conv (residual)
    - 'window' of feedback states, each is a vector of size feedback_dim
    - Summation of conv-out + feedback-out => produce next-token logits
    - We take the final conv hidden state => project => feedback vector
    """

    def __init__(self, vocab_size, embed_size, n_layers, channels, ksz, w_sz, fb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = w_sz
        self.fb_dim = fb_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.input_proj = nn.Conv1d(embed_size, channels, kernel_size=1)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels, ksz, dilation=2**i) for i in range(n_layers)]
        )
        self.post_conv = nn.Conv1d(channels, channels, 1)

        # Project flattened feedback vector to channel dim
        self.feedback_fc = nn.Linear(w_sz * fb_dim, channels)

        # Output projection => logits
        self.output_proj = nn.Conv1d(channels, vocab_size, 1)

        # final hidden => feedback vector
        self.fb_extractor = nn.Linear(channels, fb_dim, bias=False)

    def init_fb_window(self, bsz, device):
        """Initialize feedback window to zeros: [B, w_sz, fb_dim]."""
        return torch.zeros(bsz, self.window_size, self.fb_dim, device=device)

    def update_fb_window(self, fb_window, new_vec):
        """Shift left by 1, place new_vec at the right end."""
        return torch.cat([fb_window[:, 1:], new_vec.unsqueeze(1)], dim=1)

    def forward(self, x, fb_window):
        """
        x: [B, T]  (T can be 1 if step-by-step, or >1 if we feed multiple tokens)
        fb_window: [B, w_sz, fb_dim]

        Returns:
          logits      -> [B, T, vocab_size]
          final_hidden-> [B, channels] from the last time-step
        """
        B, T = x.shape

        # 1) Embed
        emb = self.embedding(x)  # [B, T, embed_size]
        emb = emb.permute(0, 2, 1)  # [B, embed_size, T]
        c = self.input_proj(emb)  # [B, channels, T]

        # 2) Residual causal blocks
        for blk in self.blocks:
            c = blk(c)
        c = self.post_conv(c)  # [B, channels, T]

        # 3) Merge feedback window
        fb_flat = fb_window.view(B, -1)  # [B, w_sz*fb_dim]
        fb_feats = self.feedback_fc(fb_flat)  # [B, channels]
        fb_feats = fb_feats.unsqueeze(-1)  # [B, channels, 1]
        fb_feats = fb_feats.expand(-1, -1, T)  # [B, channels, T]
        combined = c + fb_feats

        # 4) Project to vocab => [B, T, vocab_size]
        out = self.output_proj(combined)
        logits = F.log_softmax(out.permute(0, 2, 1), dim=-1)

        # 5) final_hidden = last time-step's hidden => [B, channels]
        final_hidden = combined[:, :, -1]
        return logits, final_hidden

    def feedback_from_hidden(self, hidden):
        """
        Convert final hidden => a feedback vector.
        This is part of the forward graph, so we do NOT wrap in no_grad().
        """
        return self.fb_extractor(hidden)


##############################################################################
#                      TRAINING (step-by-step, no dummy tokens)
##############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveFeedbackModel(
    vocab_size, embed_size, num_layers, channels, kernel_size, window_size, feedback_dim
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.NLLLoss()

seq_tensor = seq_tensor.to(device)
fb_window = model.init_fb_window(bsz=B, device=device)

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()

    # Re-init feedback each epoch (or let it carry over)
    fb_window = model.init_fb_window(bsz=B, device=device)

    # We'll train in a step-by-step manner:
    #   at step i, we feed token i, predict token i+1
    #   (so we only train up to seq_len-1, the last step has no "next token")
    for i in range(seq_len - 1):
        optimizer.zero_grad()

        # 1) Input token is the actual token from the text at position i
        inp = seq_tensor[:, i : i + 1]  # shape [1, 1]

        # 2) Forward pass
        logits, final_h = model(inp, fb_window)

        # 3) feedback vector
        new_fb_vec = model.feedback_from_hidden(final_h)

        # 4) The target is the next token in the sequence
        target = seq_tensor[:, i + 1]  # shape [1]

        loss = loss_fn(logits[:, -1, :], target)
        loss.backward()

        # 5) Clip + step
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        # 6) Detach feedback vector
        new_fb_vec = new_fb_vec.detach()

        # 7) update the feedback window
        fb_window = model.update_fb_window(fb_window, new_fb_vec)

    avg_loss = total_loss / (seq_len - 1)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss per char: {avg_loss:.4f}")


##############################################################################
#                              GENERATION
##############################################################################
def generate(model, length=60, start_token=None):
    """
    Autoregressive generation.
    If start_token is None, we'll default to the first char from the training text.
    Otherwise, pass an int [0..255].
    """
    model.eval()
    device = next(model.parameters()).device
    fb_window = model.init_fb_window(bsz=1, device=device)

    out_chars = []

    # If user doesn't provide a start token, let's pick the first char from the text
    if start_token is None:
        start_token = seq_tensor[0, 0].item()  # first char of training text

    curr_token = start_token

    with torch.no_grad():
        for _ in range(length):
            # feed the current token
            inp = torch.tensor([[curr_token]], device=device)
            logits, final_h = model(inp, fb_window)

            # feedback update
            new_fb_vec = model.feedback_from_hidden(final_h)
            fb_window = model.update_fb_window(fb_window, new_fb_vec)

            # pick next token (argmax or sample)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            out_chars.append(next_token)

            # next iteration, feed the predicted token
            curr_token = next_token

    # Convert IDs to ASCII
    return "".join(chr(c) if 0 <= c < 256 else "?" for c in out_chars)


print("\nGenerated text:")
print(generate(model, 80))
