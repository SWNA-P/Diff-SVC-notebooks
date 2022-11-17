Diff-SVC walk through, translation by [Nekro](https://twitter.com/NekroTheCorpse)

# Diff-SVC(train/inference by yourself)
## 0. Setting up the environment
>Notice: The requirements files have been updated and there are now three versions to choose from.  
1. requirements.txt installs the full environment during beta testing, it includes Torch1.12.1+cu113, you can use pip to remove the files regarding pytorch and use your own torch environment.
```
pip install -r requirements.txt
```
>2. (Recommeded): requirements_short.txt is a manually organized version of the one mentioned above that doesn't include torch itself, you can also just run the code below:
```
pip install -r requirements_short.txt
```
>3. In the root directory there's a list of requirements made by @三千 (requirements.png), it was tested on a cloud server, the torch version however, isn't compatible with the current version of torch, but the versions of the other requirements can still be taken into consideration. Thank you

## 1.Inference
>You can run inferencing with inference.ipynb in the root directory or use infer.py created by the poster @小狼\
edit the parameters mentioned in the first block:
```
config_path= 'location of config.yaml in the package'
#EX: './checkpoints/nyaru/config.yaml'
#The config and checkpoints go hand in hand, please refrain from using other configs.

project_name='current project name'
#EX: 'nyaru'

model_path='full path of the ckpt file'
#EX: './checkpoints/nyaru/model_ckpt_steps_112000.ckpt'

hubert_gpu=True
#Whether or not the GPU is used for inferencing the hubert module(a module in the model), it won't affect any other parts of the model.  The current version greatly decreases the GPU usage for inferencing the hubert module. As full inferencing can be carried out on even a 6G 1060, there is no need to turn it off.
Also, there is currently support for splicing long samples (both inference.ipynb and infer.py are capable of this), samples that are over 30 seconds are automatically spliced at silences to render, thank @小狼 for their code contribution.

```
### adjustable parameters：
```
wav_fn='xxx.wav'  
#path of input audio, default path is located in root directory

use_crepe=True  
#Crepe is a F0 calculation algorithm, it's good but slow, setting the value to False will change the F0 calculation algorithm from crepe to parselmouth that is faster than crepe but is of lower quality

thre=0.05  
#Crepe's noise filter threshold, you can increase the value of the raw audio is clean, and if there is a lot of noise, you can keep or decrease the value, changing the previous parameter to False will disable this parameter.

pndm_speedup=20  
#The multiple of the inference acceleration , the default value is 1000 steps, inputting a value if 10 would mean only using 100 steps to render, it's a rather straightforward value. The value can go up to 50x (rendering in 20 steps) without causing audible quality loss, if the value is set any higher it may start to cause quality loss. Note that if use_gt_mel is set to True below, you should keep this value lower than the add_noise_step value and keep it at a value where it can completely divide 1000.

key=0
#Key changing parameter, the default value is 0 (NOT 1!!), this shifts the raw audio up by one semitone before rendering, if the raw input is of a male voice and the desired voice is female, you can input 8 or 12 etc (12 would shift a whole octave).

use_pe=True
#F0 extraction algorithm for MEL spectogram rendering, using False will use the raw input's F0 for rendering. There's usually a difference in output between using True and False for rendering, usually setting it to True yields better results, but it's not set in stone, either value doesn't impact rendering speeds much. (Whatever the key value is, this is always changeable, doesn't affect it)

use_gt_mel=False
#This option is similar to the image-to-inage function of AI art generation, if set to True, the output audio shall be a mix of the input voice and the target voice, the percentage of each is decided by the next parameter.
NOTE!!!: If this parameter is set to true, keep the key parameter value at 0, as rendering with various pitch input is not supported.

add_noise_step=500
#Related to the previous parameter, it controls the balance of the input and target voice, a value of 1 is completely the raw input, a value of 1000 is completely the target voice, there's an audible mix in tone when the value falls around 300 (this value isn't linear, also, if this parameter is set very low, you can decrease the pndm exceleration value for higher rendering quality)

wav_gen='yyy.wav'#path of output audio, default oath is located in the root directory
```
If infer.py is used, the editing process is similar, change out the '__main__' part in __name__== then run \python infer.py\ in the root directory.  
This method requires the raw input to be put under raw and the output will be found under results.

## 2.Data preparation and training
### 2.1 Data prep
>Currently both wav and ogg format audio is supported, it's best that the sampling rate be above 24kHz, the program will automatically deal with issues regarding sampling rate and recording channels. The sampling rate can not be under 16kHz (it usually won't)
The audio should to be spliced into short samples between 5 to 15 seconds long, while there is no limit to the audio length, it's best that it not be too long or too short. The audio should be clean acapellas of the target voice, there should be no background music or the voice of other people, it's best that there be no background noise at all. If the audio is an extracted acapella, please keep the audio quality as high as possible.
Currently only single voice models are supported, the overall audio length should be at 5 hours or above, no labeling is required, it just has to be in the raw_data_dir under the root directory, how the audio is organized under this directory is completely up to you, the program can locate the required files itself.

### 2.2 Editing extra parameters
>First make a copy of config.yaml, then edit it:
The parameters below might be used (the project name will use nyaru as an example) :
```
K_step: 1000
#The step amount during diffusion, changing this is not recommended

binary_data_dir: data/binary/nyaru
#The location of the pre-processed data: the last part needs to be changed to the current project name

config_path: training/config.yaml
#The path of the config.yaml you're using, due to the fact that the pre-processing process will write in data, the path here should be a full path and not a relative path.

choose_test_manually: false
#Manually selecting an evaluation group, the default is false, and will automatically grab 5 samples for evaluation.
#If set to true, input the file name prefixes of the samples you want to use for evaluation, you can have multiple prefix entries in one list like so:

test_prefixes:
- test
- aaaa
- 5012
- speaker1024
IMPORTANT: the evaluation list CANNOT be empty, to prevent unexpected errors, it's recommended to just leave it at false.

endless_ds:False
#If your dataset is too small, each epoch will pass very fast, setting this to True will calculate 1000 epochs as a single one.

hubert_path: checkpoints/hubert/hubert.pt
#The location of the hubert module, make sure that the path is correct, in most cases, the decompressed checkpoints would be in this directory so edits won't be needed, the torch version is currently used for inferencing

hubert_gpu:True
#Whether or not the GPU is used for run the hubert model during pre-processing, CPU is used if set to False, but the required time will also increase notably. Additionally, whether the GPU is used during inferencing after training is controlled in the inference stage and is not affected by this. Currently, after the hubert version was changed to the torch version, it's possible to run pre-processing on a 6G 1060 GPU and render samples under 1 minute without exceeding vram limits, so it's normally unnecessary to set to False.

lr: 0.0008
#The original learning rate, the current value corresponds to the batch size of 88, if the batch size is lower, you can decrease the lr value accordingly.

decay_steps: 20000
The lr is halved every 20000 steps, if the batch size is lower, please increase this value.

#For a batch size of 30 to 40, the recommended values are:
lr=0.0004，decay_steps=40000

max_frames: 42000
max_input_tokens: 6000
max_sentences: 88
max_tokens: 128000
#The batch size is calculated from these values, if you're not exactly sure with what these values mean, you can edit the max_sentences value on it's own to set the max batch size to prevent exceeding vram limits.

pe_ckpt: checkpoints/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt
#Path of the pe model, make sure that this file exists, reference the inference section for the exact use of it.

raw_data_dir: data/raw/nyaru
#Directory of the raw data before pre-processing, please place the raw wav/ogg data under this directory, the organization structure inside this doesn't matter.

speaker_id: nyaru
#As the name states, currently only single voices are supported, input the name here.

use_crepe: true
#Use crepe to extract F0 for pre-processing, keep at true for higher quality, set to false for speed.

val_check_interval: 2000
#Renders the evaluation list and creates a checkpoint every 2000 steps.

work_dir: checkpoints/nyaru
#The last part is the project name
```
>Don't edit the other parameters if you don't know that they do.

### 2.3 Data pre-processing
run the following command under the diff-svc directory: \
#windows
```
set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0
python preprocessing/binarize.py --config training/config.yaml
```
#linux
```
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
For pre-processing, @小狼 made a code for processing hubert and other features in batches as doing it normally would cause you to run out if vram easily, you can run python ./network/hubert/hubert_model.py then run pre-processing  afterwards.

### 2.4 Training
#windows
```
set CUDA_VISIBLE_DEVICES=0
python run.py --config training/config.yaml --exp_name nyaru --reset 
```
#linux
```
CUDA_VISIBLE_DEVICES=0 python run.py --config training/config.yaml --exp_name nyaru --reset
```
>you need to change the exp_name to your project name and edit the config path, make sure that it's the same one as the one used for pre-processing\
*IMPORTANT* ：After training is finished, if training is not done locally, not only do you need to download the ckpt files but also the config files. As some data is written into the config during pre-processing, the config for  pre-processing and inferencing must be the same.

### 2.5 Possible issues：
>2.5.1 'Upsample' object has no attribute 'recompute_scale_factor'\
This problem is found in the torch version for cuda 11.3, if this issue occurs, please find the torch.nn.modules.upsampling.py file in your python package (for example, in a conda environment, it's located under conda_dir\envs\environment_dir\Lib\site-packages\torch\nn\modules\upsampling.py)，edit the 153-154 lines
```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,recompute_scale_factor=self.recompute_scale_factor)
```
>change to
```
return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
# recompute_scale_factor=self.recompute_scale_factor)
```
>2.5.2 no module named 'utils'\
In your computing environment (such as colab notebooks), set it up like so:
```
import os
os.environ['PYTHONPATH']='.'
!CUDA_VISIBLE_DEVICES=0 python preprocessing/binarize.py --config training/config.yaml
```
Note that this must be done in the root directory.
>2.5.3 cannot load library 'libsndfile.so'\
This is an error that can occur in a Linux environment, please run the following code:
```
apt-get install libsndfile1 -y
```
>2.5.4 cannot load import 'consume_prefix_in_state_dict_if_present'\
The version of torch is too old, please upgrade to a higher version.

>2.5.5 Data pre-processing being too slow\
Check if use_crepe is enabled in the settings, turning it off can greatly increase speeds.\
Check if hubert_gpu is enabled in settings.

If there are any other question, please scan the QR code at the bottom of this repository for inquiry. (T/N: It's a Chinese group on QQ, so do take notice of that)

<img src="https://user-images.githubusercontent.com/107520869/202339768-68ab0dae-9871-406e-a393-07b5b157b0c2.jpg" width="200">

(T/N: For English support, you can contact Nekro via discord via username (a new day a new disappointment#9085), or twitter (@NekroTheCorpse).
