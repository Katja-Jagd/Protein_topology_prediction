import torch.nn as nn
import torch
import torch.nn.functional as F

class Model1(nn.Module):
  
    def __init__(self):
        super(Model1, self).__init__()
        self.input_size = 512
        self.hidden_size = 64
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 6)  # Output size set to 6 classes per label
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # Process the input data
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x


class Model2(nn.Module):
    def __init__(self, device):
        super(Model2, self).__init__()
        self.input_size = 512
        self.hidden_size = 64
        self.num_layers = 2
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, device = device)
        self.lstm2 = nn.LSTM(self.hidden_size, 6, num_layers=self.num_layers, batch_first=True, bidirectional=False, device = device)
        self.fc = nn.Linear(6, 6)  # Fully connected layer

    def forward(self, x):
        # Process the input data
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        x = self.fc(out2)

        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, encoder_outputs):
        # Calculate attention scores
        attn_scores = torch.matmul(encoder_outputs, self.attn_weights)
        attn_scores = torch.matmul(attn_scores, encoder_outputs.transpose(1, 2))
        attn_scores = F.softmax(attn_scores, dim=2)  # Apply softmax along the sequence_length dimension

        # Apply attention weights to encoder outputs
        attn_outputs = torch.matmul(attn_scores, encoder_outputs)
        return attn_outputs, attn_scores

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.input_size = 512
        self.hidden_size = 128
        self.output_size = 6
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.activation = nn.ReLU()
        self.attention = Attention(self.hidden_size)

    def forward(self, x):
        # reduce dimensionality of mask from [batch_size, sequence_length, 512] to [batch_size, sequence_length]
        mask = mask[:, :, 0]

        # Apply the mask to ignore padded elements
        x = x * mask.unsqueeze(-1)
        output, _ = self.lstm(x)
        output = self.activation(output)

        # print("output shape before attention:", output.shape)


        # Apply attention mechanism
        attn_output, attn_scores = self.attention(output)

        # print("output shape after attention:", attn_output.shape)


        # Fully connected layer
        output = self.linear(attn_output)
        output = self.activation(output)

        # Pass through linear output layer
        output = self.out(output)

        # print("output shape after output layer:", output.shape)


        return output


class Model4(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, num_classes=6, dropout_prob=0.5):
        super(Model4, self).__init__()
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Dense layers with dropout
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Bidirectional LSTM, so multiply hidden_size by 2
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM layer
        lstm_output, _ = self.lstm(x)
        
        # Feedforward through dense layers
        x = self.fc1(lstm_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
