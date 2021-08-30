import tensorflow as tf

def split_heads(x, num_heads):
    batch_size = tf.shape(x)[0]
    input_len  = tf.shape(x)[1]
    depth_size = tf.cast(
        tf.shape(x)[2] / num_heads, tf.int32)
    
    split_outputs = tf.reshape(
        x, [batch_size, input_len, num_heads, depth_size])
    return tf.transpose(split_outputs, [0, 2, 1, 3])

def combine_heads(x):
    batch_size = tf.shape(x)[0]
    input_len  = tf.shape(x)[2]
    num_heads  = tf.shape(x)[1]
    depth_size = tf.shape(x)[3]
    hidden_size = num_heads*depth_size
    
    combined_outputs = tf.reshape(tf.transpose(
        x, [0, 2, 1, 3]), [batch_size, input_len, hidden_size])
    return combined_outputs

def layer_normalisation(x, bias, scale, eps=1.0e-6):
    x_mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    x_var  = tf.reduce_mean(
        tf.square(x - x_mean), axis=[-1], keepdims=True)
    x_norm = (x - x_mean) * tf.math.rsqrt(x_var + tf.constant(eps))
    return (x_norm * scale) + bias

class GPT_Network(tf.keras.Model):
    def __init__(
    self, n_layers, n_heads, 
    hidden_size, ffwd_size, vocab_size, seq_length, 
    win_length, embed_size=128, p_keep=0.9, p_reg=1.0, 
    var_type="norm_add", attn_type="mult_attn", **kwargs):
        super(GPT_Network, self).__init__(**kwargs)
        self.p_keep = p_keep
        self.n_heads  = n_heads
        self.n_layers = n_layers
        self.var_type  = var_type
        self.attn_type = attn_type
        
        self.seq_length = seq_length
        self.win_length = win_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.ffwd_size  = ffwd_size
        self.head_size  = int(hidden_size / n_heads)
        self.hidden_size = hidden_size
        
        # Embedding matrices. #
        emb_shape = [self.vocab_size, self.embed_size]
        lin_shape = [self.embed_size, self.hidden_size]
        
        self.W_dec_lin = tf.Variable(tf.random.normal(
            lin_shape, stddev=0.1), name="dec_linear")
        self.W_emb_dec = tf.Variable(tf.random.normal(
            emb_shape, stddev=0.1), name="dec_embedding")
        
        # Output projection. #
        logits_shape = [self.hidden_size, self.vocab_size]
        self.p_decoder = tf.Variable(tf.random.normal(
            logits_shape, stddev=0.1), name="p_decoder")
        
        # GPT Variables. #
        param_norm_shp  = [self.n_layers, self.hidden_size]
        attn_add_shape  = [self.n_layers, self.head_size, 1]
        attn_wgt_shape  = [
            self.n_layers, self.hidden_size, self.hidden_size]
        attn_ffw1_shape = [
            self.n_layers, self.hidden_size, self.ffwd_size]
        attn_ffw2_shape = [
            self.n_layers, self.ffwd_size, self.hidden_size]
        pos_embed_shape = [
            self.n_layers, self.win_length*2, self.hidden_size]
        
        self.p_d_q = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_q")
        self.p_d_k = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_k")
        self.p_d_v = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_v")
        self.p_d_c = tf.Variable(tf.random.normal(
            attn_wgt_shape, stddev=0.1), name="p_d_c")
        
        if self.attn_type == "add_attn":
            self.v_d_s = tf.Variable(tf.random.normal(
                attn_add_shape, stddev=0.1), name="v_d_s")
            self.v_c_s = tf.Variable(tf.random.normal(
                [self.head_size, 1], stddev=0.1), name="v_c_s")
        
        self.p_d_ff1 = tf.Variable(tf.random.normal(
            attn_ffw1_shape, stddev=0.1), name="p_d_ff1")
        self.p_d_ff2 = tf.Variable(tf.random.normal(
            attn_ffw2_shape, stddev=0.1), name="p_d_ff2")
        self.b_d_ff1 = tf.Variable(tf.zeros(
            [self.n_layers, self.ffwd_size]), name="b_d_ff1")
        self.b_d_ff2 = tf.Variable(tf.zeros([
            self.n_layers, self.hidden_size]), name="b_d_ff2")
        
        self.b_d_bias_1 = tf.Variable(
            tf.zeros(param_norm_shp), name="b_d_bias_1")
        self.b_d_bias_2 = tf.Variable(
            tf.zeros(param_norm_shp), name="b_d_bias_2")
        self.b_d_scale_1 = tf.Variable(
            tf.ones(param_norm_shp), name="b_d_scale_1")
        self.b_d_scale_2 = tf.Variable(
            tf.ones(param_norm_shp), name="b_d_scale_2")
        
        # Position Embeddings. #
        self.x_emb_pos_dec = tf.Variable(tf.random.normal(
            pos_embed_shape, stddev=0.1), name="pos_embed")
    
    def transformer_decode(
        self, step, dec_inputs, p_reg=1.0, training=False):
        if training:
            p_keep = self.p_keep
        else:
            p_keep = 1.0
        batch_size = tf.shape(dec_inputs)[0]
        
        if step <= self.win_length:
            n_blocks = 1
        elif step % self.win_length == 0:
            n_blocks = int(step / self.win_length)
        else:
            n_blocks = int(step / self.win_length) + 1
        
        output_list = []
        prev_inputs = tf.zeros([
            self.n_layers, batch_size, 
            self.win_length, self.hidden_size])
        
        for n_block in range(n_blocks):
            id_st = n_block * self.win_length
            if n_block == (n_blocks-1):
                id_en = step
            else:
                id_en = (n_block+1) * self.win_length
            
            if n_block == 1:
                decode_step = id_en - id_st
                x_embed_pos = self.x_emb_pos_dec[:, :id_en, :]
            elif n_block == (n_blocks-1):
                decode_step = id_en - id_st + self.win_length
                
                if step % self.win_length == 0:
                    x_embed_pos = self.x_emb_pos_dec
                else:
                    res_len = (step % self.win_length)
                    pos_len = self.win_length + res_len
                    x_embed_pos = self.x_emb_pos_dec[:, :pos_len, :]
            else:
                decode_step = id_en - id_st + self.win_length
                x_embed_pos = self.x_emb_pos_dec
            
            # Decode the output using Transformer. #
            tmp_inputs = dec_inputs[:, id_st:id_en, :]
            tmp_outputs = self.base_decode(
                decode_step, x_embed_pos, 
                prev_inputs, tmp_inputs, p_keep, p_reg=p_reg)
            
            prev_inputs = tmp_outputs
            output_list.append(tmp_outputs[-1, :, :, :])
        
        # Concatenate into a tensor. #
        dec_outputs = tf.concat(output_list, axis=1)
        return dec_outputs
    
    def base_decode(
        self, step, x_emb_pos, 
        prev_inputs, dec_inputs, p_keep, p_reg=1.0):
        n_heads = self.n_heads
        l_input = tf.shape(dec_inputs)[1]
        head_size = tf.cast(self.head_size, tf.float32)
        
        layer_out = []
        neg_infty = -1.0e9
        mask_len  = l_input + self.win_length
        attn_triu = tf.linalg.band_part(
            tf.ones([mask_len, mask_len]), -1, 0)
        
        attn_mask = neg_infty * (1.0 - attn_triu)
        attn_mask = tf.expand_dims(
            tf.expand_dims(attn_mask, axis=0), axis=0)
        
        layer_input = dec_inputs
        for m in range(self.n_layers):
            conc_in = tf.concat(
                [prev_inputs[m, :, :, :], layer_input], axis=1)
            layer_in = tf.add(x_emb_pos[m, :, :], conc_in)
            
            # Self Attention Layer. #
            x_sq = split_heads(tf.tensordot(
                layer_in, self.p_d_q[m], [[2], [0]]), n_heads)
            x_sq = x_sq * tf.math.rsqrt(head_size)
            
            x_sk = split_heads(tf.tensordot(
                layer_in, self.p_d_k[m], [[2], [0]]), n_heads)
            x_sv = split_heads(tf.tensordot(
                layer_in, self.p_d_v[m], [[2], [0]]), n_heads)
            
            if self.attn_type == "add_attn":
                x_sq = tf.expand_dims(x_sq, axis=2)
                x_sk = tf.expand_dims(x_sk, axis=3)
                
                x_s_scores = tf.tensordot(tf.nn.tanh(
                    x_sq + x_sk), self.v_d_s[m], [[4], [0]])
                x_s_scores = tf.transpose(tf.squeeze(
                    x_s_scores, axis=4), [0, 1, 3, 2])
                x_s_alphas = tf.nn.softmax(x_s_scores + attn_mask)
            else:
                x_s_scores = tf.matmul(
                    x_sq, x_sk, transpose_b=True)
                x_s_alphas = \
                    tf.nn.softmax(x_s_scores + attn_mask)
            
            x_self_conc = tf.matmul(x_s_alphas, x_sv)
            x_self_conc = combine_heads(x_self_conc)
            
            x_multi_self = tf.tensordot(
                x_self_conc, self.p_d_c[m], [[2], [0]])
            x_multi_self = tf.nn.dropout(
                x_multi_self, rate=1.0-p_reg)
            
            tmp_bias1  = self.b_d_bias_1[m]
            tmp_scale1 = self.b_d_scale_1[m]
            if self.var_type == "norm_add":
                x_self_norm = tf.add(
                    layer_in, layer_normalisation(
                        x_multi_self, tmp_bias1, tmp_scale1))
            elif self.var_type == "add_norm":
                x_self_norm = layer_normalisation(tf.add(
                    layer_in, x_multi_self), tmp_bias1, tmp_scale1)
            
            # Feed forward layer. #
            x_ffw1 = tf.nn.relu(tf.add(
                self.b_d_ff1[m], tf.tensordot(
                    x_self_norm, self.p_d_ff1[m], [[2], [0]])))
            x_ffw2 = tf.add(
                self.b_d_ff2[m], tf.tensordot(
                    x_ffw1, self.p_d_ff2[m], [[2], [0]]))
            x_ffwd = tf.nn.dropout(x_ffw2, rate=1.0-p_keep)
            
            tmp_bias2  = self.b_d_bias_2[m]
            tmp_scale2 = self.b_d_scale_2[m]
            if self.var_type == "norm_add":
                x_ffw_norm = tf.add(
                    x_self_norm, layer_normalisation(
                        x_ffwd, tmp_bias2, tmp_scale2))
            elif self.var_type == "add_norm":
                x_ffw_norm = layer_normalisation(tf.add(
                    x_self_norm, x_ffwd), tmp_bias2, tmp_scale2)
            
            # Append the output. #
            layer_input = x_ffw_norm[:, -l_input:, :]
            layer_out.append(tf.expand_dims(
                x_ffw_norm[:, -l_input:, :], axis=0))
        
        dec_outputs = tf.concat(layer_out, axis=0)
        return dec_outputs
    
    def call(self, x_input, training=True):
        # Word or Sub-word embeddings. #
        x_dec_token = tf.nn.embedding_lookup(
            self.W_emb_dec, x_input)
        
        # Transformer Decoder. #
        x_dec_embed = tf.tensordot(
            x_dec_token, self.W_dec_lin, [[2], [0]])
        
        # Training via Teacher forcing. #
        dec_outputs = self.transformer_decode(
            self.seq_length, x_dec_embed, training=training)
        
        dec_logits = tf.tensordot(
            dec_outputs, self.p_decoder, [[2], [0]])
        return dec_logits
    
    def compute_xent_loss(
        self, x_inputs, x_labels, p_reg=1.0):
        step = self.seq_length
        p_keep = self.p_keep
        batch_size = tf.shape(x_inputs)[0]
        
        if step <= self.win_length:
            n_blocks = 1
        elif step % self.win_length == 0:
            n_blocks = int(step / self.win_length)
        else:
            n_blocks = int(step / self.win_length) + 1
        
        tot_losses  = 0.0
        prev_inputs = tf.zeros([
            self.n_layers, batch_size, 
            self.win_length, self.hidden_size])
        
        for n_block in range(n_blocks):
            id_st = n_block * self.win_length
            if n_block == (n_blocks-1):
                id_en = step
            else:
                id_en = (n_block+1) * self.win_length
            
            if n_block == 1:
                decode_step = id_en - id_st
                x_embed_pos = self.x_emb_pos_dec[:, :id_en, :]
            elif n_block == (n_blocks-1):
                decode_step = id_en - id_st + self.win_length
                
                if step % self.win_length == 0:
                    x_embed_pos = self.x_emb_pos_dec
                else:
                    res_len = (step % self.win_length)
                    pos_len = self.win_length + res_len
                    x_embed_pos = self.x_emb_pos_dec[:, :pos_len, :]
            else:
                decode_step = id_en - id_st + self.win_length
                x_embed_pos = self.x_emb_pos_dec
            
            # Decode the output using Transformer. #
            tmp_embed  = tf.nn.embedding_lookup(
                self.W_emb_dec, x_inputs[:, id_st:id_en])
            tmp_inputs = tf.tensordot(
                tmp_embed, self.W_dec_lin, [[2], [0]])
            tmp_labels = x_labels[:, id_st:id_en]
            
            tmp_outputs = self.base_decode(
                decode_step, x_embed_pos, 
                prev_inputs, tmp_inputs, p_keep, p_reg=p_reg)
            
            prev_inputs  = tmp_outputs
            curr_outputs = tmp_outputs[-1, :, :, :]
            curr_logits  = tf.tensordot(
                curr_outputs, self.p_decoder, [[2], [0]])
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_labels, logits=curr_logits), axis=1))
            tot_losses += tmp_losses
        return tot_losses
    
    def infer(self, x_infer):
        # Inference. #
        infer_len = tf.shape(x_infer)[1]
        x_inf_emb = tf.nn.embedding_lookup(self.W_emb_dec, x_infer)
        infer_emb = [tf.expand_dims(x_inf_emb[:, 0, :], axis=1)]
        infer_ids = [tf.expand_dims(x_infer[:, 0], axis=1)]
        
        for step in range(self.seq_length):
            x_inf_inputs = tf.concat(infer_emb, axis=1)
            x_inf_inputs = tf.tensordot(
                x_inf_inputs, self.W_dec_lin, [[2], [0]])
            
            tmp_outputs = self.transformer_decode(
                step+1, x_inf_inputs)
            
            tmp_logit  = tf.matmul(
                tmp_outputs[:, -1, :], self.p_decoder)
            tmp_argmax = tf.cond(
                step < (infer_len-1), 
                lambda: x_infer[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            next_embed = tf.cond(
                step < (infer_len-1), 
                lambda: x_inf_emb[:, step+1, :], 
                lambda: tf.matmul(
                    tf.nn.softmax(tmp_logit), self.W_emb_dec))
            
            infer_ids.append(tf.expand_dims(tmp_argmax, axis=1))
            infer_emb.append(tf.expand_dims(next_embed, axis=1))
        
        infer_indices = tf.concat(infer_ids, axis=1)
        return infer_indices
