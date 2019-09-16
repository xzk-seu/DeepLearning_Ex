import torch
from torch import nn


torch.manual_seed(1)    # reproducible


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=28,          # 图片宽度，每一步输入的大小
            hidden_size=64,         # rnn hidden unit
            num_layers=2,           # number of rnn layer 多层的话，上层的隐层的输出作为下层的输入
            batch_first=True,       # input & output will has batch size as 1s dimension.
                                    # e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, hidden_size) 每个时间步上，隐层都有输出
        # h_n shape (n_layers, batch, hidden_size) # 最后一步隐层的输出
        # c_n shape (n_layers, batch, hidden_size)
        r_out, (h_n, c_n) = self.rnn(x, None)   # None represents zero initial hidden state

        """
        # r_out为每个时间步上的输出， h_n为最后一个时间步的隐层输出， c_n为最后一步cell_state
        t_1 = h_n[-1, :, :]
        t_2 = r_out[:, -1, :]
        """

        # choose r_out at the last time step
        # 把最后一步的结果用于生成最终结果
        out = self.out(r_out[:, -1, :])
        return out, r_out[:, -1, :]


if __name__ == '__main__':
    rnn = RNN()
    print(rnn)
