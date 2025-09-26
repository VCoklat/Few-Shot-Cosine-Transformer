Train phase: 
  0%|                                                   | 0/200 [00:05<?, ?it/s]
Traceback (most recent call last):
  File "/kaggle/working/Few-Shot-Cosine-Transformer/train_test.py", line 344, in <module>
    model = train(base_loader, val_loader, model, optimization, params.num_epoch, params)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Few-Shot-Cosine-Transformer/train_test.py", line 133, in train
    model.train_loop(epoch, num_epoch, base_loader, params.wandb, optimizer)
  File "/kaggle/working/Few-Shot-Cosine-Transformer/methods/meta_template.py", line 70, in train_loop
    acc, loss = self.set_forward_loss(x = x.to(device))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Few-Shot-Cosine-Transformer/methods/transformer.py", line 73, in set_forward_loss
    cov_loss = covariance_regularization(embeddings)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/Few-Shot-Cosine-Transformer/methods/transformer.py", line 17, in covariance_regularization
    cov = (centered.t() @ centered) / m
           ^^^^^^^^^^^^
RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D