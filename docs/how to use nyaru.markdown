Hi! **Obama Care** here, and this is a tutorial on how to use nyaru (the pre-included model) as a means for a reduced training time.

>This guide assumes that you **already** know how Diff-SVC works and how to train a model, alongside basic stuff. This applies to local training and Google Colab training.

## Start

 1. Get the nyaru base. You should know where this is. If not then it's mirrored in my fork.
 2. Put the latest checkpoint into your model folder.
 3. **Set lr to roughly 0.008 or 0.0005.**
 4. **Set decay to about 150k.**
 5. Start training. It should be legible at around the 120k steps mark. More training is likely required if you do not have the recommended amount of data or if your database is a male voice type. You'll know it worked when it says resuming training from checkpoint.

**Not enough training can cause the model to sound very off compared to the source database.**

![obama real???](https://beebom.com/wp-content/uploads/2022/10/Chainsaw-Man-Who-is-Pochita-All-You-Need-to-Know.jpg)
