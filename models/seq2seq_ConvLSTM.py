import torch
import torch.nn as nn
import random
from models.ConvLSTMCell import ConvLSTMCell

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=256,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=256,
                                               hidden_dim=128,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_3_convlstm = ConvLSTMCell(input_dim=128,
                                               hidden_dim=64,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_4_convlstm = ConvLSTMCell(input_dim=64,
                                               hidden_dim=64,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv2d(in_channels=65,
                                     out_channels=out_chan,
                                     kernel_size=(3, 3),
                                     padding=(0, 0))

        self.relu = nn.ReLU(inplace=False)


    def autoencoder(self, x, x0t, m, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []
        emputs = []

        # encoder
        for t in range(seq_len):
            if( t>0 and random.random() < 1.1):
                encoder_vector = torch.cat((x[:, t, :, :, :],x0,m),1)
            else:
                encoder_vector = torch.cat((x[:, t, :, :, :],x0t[:, t, :, :, :],m),1)
            h_t, c_t = self.encoder_1_convlstm(input_tensor=encoder_vector,
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            h_t3, c_t3 = self.encoder_3_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t3, c_t3])
            h_t4, c_t4 = self.encoder_4_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])
            if( t>0 and random.random() < 1.1):
                decoder_vector = torch.cat((h_t4,x0),1)
            else:
                decoder_vector = torch.cat((h_t4,x0t[:,t,  :, :, :]),1)

            xx = self.relu(self.decoder_CNN(decoder_vector))
            x0 = x0t[:, t+1].clone()
            x0[:,:,1:-1,1:-1] = xx

            #x0 = torch.nn.Sigmoid()(x0)
            #x0 = torch.where(x0>0,x0,x0*0)
            outputs += [x0]

        '''
        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions
        '''

        outputs = torch.stack(outputs, 1)
        #outputs = outputs.permute(0, 2, 1, 3, 4)
        #outputs = self.decoder_CNN(outputs)
        #outputs = torch.nn.Sigmoid()(outputs)
        #outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

    def forward(self, x, x0, m, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.encoder_4_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, x0, m, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs
