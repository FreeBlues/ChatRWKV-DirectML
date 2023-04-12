# ChatRWKV-DirectML
ChatRWKV-DirectML is for `AMD GPU[RX6600/6700/6800/6900]` + `Windows` users.

## Installation

### Clone ChatRWKV-DirectML

```
git clone https://github.com/FreeBlues/ChatRWKV-DirectML
```

### Install & Setup Python Env

Here we will use miniconda3, you need to install it first:

#### Install miniconda3

Download [Minicoda](https://docs.conda.io/en/latest/miniconda.html) and install it to your windows system.

[Miniconda3 Windows 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)    
[Miniconda3 Windows 32-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe)    

#### Create your python env: ChatRWKV-DML

```
conda create -n ChatRWKV-DML python=3.10
conda activate ChatRWKV-DML
```

#### Install pytorch and other dependencies

```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install numpy tokenizers prompt_toolkit
```
> Notice: The version of `torch` should be `1.13.1`, the `torch.directml` only support this version by now[20230412].

#### Install torch.directml

```
pip install torch.directml
```

## Download models and put it into ChatRWKV-DirectML folder

### Models

[Models 1.5B](https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/tree/main)    
[Models 3B](https://huggingface.co/BlinkDL/rwkv-4-pile-3b/tree/main)    
[Models 7B](https://huggingface.co/BlinkDL/rwkv-4-pile-7b/tree/main)    
[Models 14B](https://huggingface.co/BlinkDL/rwkv-4-pile-14b/tree/main)    

### Put it into local folder

After download the model, then put the whole folder into the `ChatRWKV-DirectML\v2\fsx\BlinkDL\HF-MODEL`

Just like below:

```
E:\Github\ChatRWKV-DirectML\v2\fsx\BlinkDL\HF-MODEL\rwkv-4-pile-1b5
└─.gitattributes
└─README.md
└─RWKV-4-Pile-1B5-20220814-4526.pth
└─RWKV-4-Pile-1B5-20220822-5809.pth
└─RWKV-4-Pile-1B5-20220903-8040.pth
└─RWKV-4-Pile-1B5-20220929-ctx4096.pth
└─RWKV-4-Pile-1B5-Chn-testNovel-done-ctx2048-20230312.pth
└─RWKV-4-Pile-1B5-EngChn-test4-20230115.pth
└─RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225.pth
└─RWKV-4-Pile-1B5-Instruct-test1-20230124.pth
└─RWKV-4-Pile-1B5-Instruct-test2-20230209.pth
└─RWKV-4b-Pile-1B5-20230217-7954.pth
```

## Run

Now you can run it:
```
# Enter the v2    
cd E:\Github\ChatRWKV\v2\   
python chat.py   
```

We use `1.5B Model` as default

## Configuration

You can load different `model` and selet different `VRAM strategy`

### Change model

In `v2\chat.py: line 65`, select the one you want to load and remove the comment, and comment all others lines, just like below:

```
args.strategy = 'privateuse1 fp32'
# args.strategy = 'privateuse1 fp32 -> cpu fp32 *10'
# args.strategy = 'privateuse1 fp32i8 *20 -> cpu fp32i8'
# args.strategy = 'privateuse1 fp32 *20 -> cpu fp32'
# args.strategy = 'cpu fp32 -> privateuse1 fp32 *10'
# args.strategy = 'privateuse1:0 fp32*25 -> privateuse1:0 fp32'
```

### Change VRAM strategy

In `v2\chat.py: line 77`, select the VRAM strategy you want to use and remove the comment, and comment all others lines, just like below:

```
elif CHAT_LANG == 'Chinese': # testNovel系列是小说模型，请只用 +gen 指令续写。Raven系列可以对话和问答（目前中文只用了小语料，更适合创意虚构）
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v7-ChnEng-20230404-ctx2048'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-3B-v7-ChnEng-20230404-ctx2048'
    # args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-done-ctx2048-20230226'
    args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-663'
```

> Notice: It should add a `.` in fornt of `/fsx/BlinkDL/`    
>> Right: `./fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225`    
>> Wrong: `/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225`    

### More VRAM strategies    
In fact, `ChatRWKV` has supported flexible `VRAM strategies`, you can use:

-   Only GPU ram    
-   Only CPU ram    
-   GPU ram + CPU ram    

More VRAM strategies you can try yourself.  

### Samples

>  My PC:  
```
CPU: AMD Ryzen 5 5600 6-Core Processor 3.50 GHz
Ram: 48G
GPU: AMD Radeon RX6600 8G
OS: Win10 x64
```

####  Load `3B` model  
-    VRAM strategy:  
```
args.strategy = 'privateuse1 fp32 *20 -> cpu fp32'
```

-    Model selection:    
```
args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-done-ctx2048-20230226'
```

With this VRAM strategy, the model partly in GPU Ram(about 7.8G), and partly in CPU Ram。

>Notice: You need to try the number before the asterisk `*` in `VRAM strategy` according to your hardware.

####  Load `7B` model  
-    VRAM strategy:  
```
args.strategy = 'privateuse1 fp32 *5 -> cpu fp32'
```

-    Model selection:    
```
args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
```

The `7B` Model is slow on my `RX6600 8G`, output one word need 1~2 seconds.

####  Load `14B` model  
-    VRAM strategy:  
```
args.strategy = 'privateuse1 fp32i8 *5 -> cpu fp32i8'
```

-    Model selection:    
```
args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
```

The `14B` Model is hard to load, I have to use `fp32i8`, and it is very very slow on my `RX6600 8G`, output one word need several minutes(>3 mins).

---

#  中文说明：

---

# ChatRWKV-DirectML
ChatRWKV-DirectML 针对使用 `AMD GPU[RX6600/6700/6800/6900]` 系列显卡的 `Windows` 用户， 可以让 `ChatRWKV` 运行在 `AMD`显卡上.

## 安装

### 克隆 ChatRWKV-DirectML

```
git clone https://github.com/FreeBlues/ChatRWKV-DirectML
```

### 安装 & 设置 Python 环境

这里使用了 miniconda3，需要先安装它

#### 安装 miniconda3

下载 [Minicoda](https://docs.conda.io/en/latest/miniconda.html) 然后安装到你的 Windows 系统上.

[Miniconda3 Windows 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)    
[Miniconda3 Windows 32-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86.exe)    

#### 新建 python 环境: ChatRWKV-DML

```
conda create -n ChatRWKV-DML python=3.10
conda activate ChatRWKV-DML
```

#### 安装 pytorch 以及其他依赖

```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install numpy tokenizers prompt_toolkit
```
> 注意: `torch` 的版本应该是 `1.13.1`, 因为 `torch.directml` 目前为止只支持这个版本[20230412].

#### 安装 torch.directml

```
pip install torch.directml
```

## 下载模型后放入 ChatRWKV-DirectML 项目文件夹

### 模型

[模型 1.5B](https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/tree/main)    
[模型 3B](https://huggingface.co/BlinkDL/rwkv-4-pile-3b/tree/main)    
[模型 7B](https://huggingface.co/BlinkDL/rwkv-4-pile-7b/tree/main)    
[模型 14B](https://huggingface.co/BlinkDL/rwkv-4-pile-14b/tree/main)    

### 把下载回来的模型目录（整个目录）放入项目文件夹中

下载模型后，把整个模型文件夹放入 `ChatRWKV-DirectML\v2\fsx\BlinkDL\HF-MODEL`

如下所示:

```
E:\Github\ChatRWKV-DirectML\v2\fsx\BlinkDL\HF-MODEL\rwkv-4-pile-1b5
└─.gitattributes
└─README.md
└─RWKV-4-Pile-1B5-20220814-4526.pth
└─RWKV-4-Pile-1B5-20220822-5809.pth
└─RWKV-4-Pile-1B5-20220903-8040.pth
└─RWKV-4-Pile-1B5-20220929-ctx4096.pth
└─RWKV-4-Pile-1B5-Chn-testNovel-done-ctx2048-20230312.pth
└─RWKV-4-Pile-1B5-EngChn-test4-20230115.pth
└─RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225.pth
└─RWKV-4-Pile-1B5-Instruct-test1-20230124.pth
└─RWKV-4-Pile-1B5-Instruct-test2-20230209.pth
└─RWKV-4b-Pile-1B5-20230217-7954.pth
```

## 运行

现在一切就绪，可以运行了:
```
# Enter the v2    
cd E:\Github\ChatRWKV\v2\   
python chat.py   
```

默认使用 `1.5B 模型`，因为它需要的存储资源最少。

## 配置

You can load different `model` and selet different `VRAM strategy`
还可以加载不同的 `模型` 和选择不同的 `VRAM 策略`

### 换模型

在 `v2\chat.py: line 65` 中, 把你选中的那一行的注释符号 `#` 去掉, 同时保持其他所有行的注释状态不变, 如下所示:

```
args.strategy = 'privateuse1 fp32'
# args.strategy = 'privateuse1 fp32 -> cpu fp32 *10'
# args.strategy = 'privateuse1 fp32i8 *20 -> cpu fp32i8'
# args.strategy = 'privateuse1 fp32 *20 -> cpu fp32'
# args.strategy = 'cpu fp32 -> privateuse1 fp32 *10'
# args.strategy = 'privateuse1:0 fp32*25 -> privateuse1:0 fp32'
```

### 换 VRAM 策略

在 `v2\chat.py: line 77` 中, 把你选中的那一行的注释符号 `#` 去掉, 同时保持其他所有行的注释状态不变, 如下所示:

```
elif CHAT_LANG == 'Chinese': # testNovel系列是小说模型，请只用 +gen 指令续写。Raven系列可以对话和问答（目前中文只用了小语料，更适合创意虚构）
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v7-ChnEng-20230404-ctx2048'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-3B-v7-ChnEng-20230404-ctx2048'
    # args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-done-ctx2048-20230226'
    args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-663'
```

> 注意:  在 `/fsx/BlinkDL/` 最前面，要加一个点 `.`   
>> 正确写法: `./fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225`    
>> 错误写法: `/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225`    

### 更多 VRAM 策略    
实际上, `ChatRWKV` 支持灵活的 `VRAM 策略`, 如下:

-   Only GPU ram    
-   Only CPU ram    
-   GPU ram + CPU ram    

你可以自己去试验更多的 VRAM 策略.  

### 例子

> 我的 PC:  
```
CPU: AMD Ryzen 5 5600 6-Core Processor 3.50 GHz
Ram: 48G
GPU: AMD Radeon RX6600 8G
OS: Win10 x64
```

#### 加载 `3B` 模型

-    VRAM 策略:  
```
args.strategy = 'privateuse1 fp32 *20 -> cpu fp32'
```

-    模型选择:    
```
args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-done-ctx2048-20230226'
```

使用这个 VRAM 策略, 一部分模型被加载到 GPU Ram(大约 7.8G)中, 一部分模型被加载到 CPU Ram 中。

> 注意: 你需要根据自己的硬件配置（主要是 GPU 的显存）来调整 `VRAM 策略` 中星号 `*` 后面的数字.

####  加载 `7B` 模型  

-    VRAM 策略:  
```
args.strategy = 'privateuse1 fp32 *5 -> cpu fp32'
```

-    模型选择:    
```
args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
```

`7B` 模型在我的 `RX6600 8G` 上有点慢, 每1~2秒能输出一个字.

####  加载 `14B` 模型     

-    VRAM 策略:  
```
args.strategy = 'privateuse1 fp32i8 *3 -> cpu fp32i8'
```

-    模型选择:    
```
args.MODEL_NAME = './fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-done-ctx2048-20230317'
```

`14B` 模型加载起来挺不容易，只能使用比较慢的 `fp32i8`, 而且它在我的 `RX6600 8G` 上非常非常慢, 输出一个字需要好几分钟(>3 分钟).

## 1.5B Successful Run log

```
(ChatRWKV-DML) E:\Github\ChatRWKV-DirectML\v2>python chat.py


ChatRWKV v2 https://github.com/BlinkDL/ChatRWKV

Chinese - privateuse1 fp32 - E:\Github\ChatRWKV-DirectML\v2/prompt/default/Chinese-2.py
Loading model - ./fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225
RWKV_JIT_ON 1 RWKV_CUDA_ON 0 RESCALE_LAYER 0

Loading ./fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225.pth ...
Strategy: (total 24+1=25 layers)
* privateuse1 [float32, float32], store 25 layers
0-privateuse1-float32-float32 1-privateuse1-float32-float32 2-privateuse1-float32-float32 3-privateuse1-float32-float32 4-privateuse1-float32-float32 5-privateuse1-float32-float32 6-privateuse1-float32-float32 7-privateuse1-float32-float32 8-privateuse1-float32-float32 9-privateuse1-float32-float32 10-privateuse1-float32-float32 11-privateuse1-float32-float32 12-privateuse1-float32-float32 13-privateuse1-float32-float32 14-privateuse1-float32-float32 15-privateuse1-float32-float32 16-privateuse1-float32-float32 17-privateuse1-float32-float32 18-privateuse1-float32-float32 19-privateuse1-float32-float32 20-privateuse1-float32-float32 21-privateuse1-float32-float32 22-privateuse1-float32-float32 23-privateuse1-float32-float32 24-privateuse1-float32-float32
emb.weight                        f32      cpu  50277  2048
blocks.0.ln1.weight               f32 privateuseone:0   2048
blocks.0.ln1.bias                 f32 privateuseone:0   2048
blocks.0.ln2.weight               f32 privateuseone:0   2048
blocks.0.ln2.bias                 f32 privateuseone:0   2048
blocks.0.att.time_decay           f32 privateuseone:0   2048
blocks.0.att.time_first           f32 privateuseone:0   2048
blocks.0.att.time_mix_k           f32 privateuseone:0   2048
blocks.0.att.time_mix_v           f32 privateuseone:0   2048
blocks.0.att.time_mix_r           f32 privateuseone:0   2048
blocks.0.att.key.weight           f32 privateuseone:0   2048  2048
blocks.0.att.value.weight         f32 privateuseone:0   2048  2048
blocks.0.att.receptance.weight    f32 privateuseone:0   2048  2048
blocks.0.att.output.weight        f32 privateuseone:0   2048  2048
blocks.0.ffn.time_mix_k           f32 privateuseone:0   2048
blocks.0.ffn.time_mix_r           f32 privateuseone:0   2048
blocks.0.ffn.key.weight           f32 privateuseone:0   2048  8192
blocks.0.ffn.receptance.weight    f32 privateuseone:0   2048  2048
blocks.0.ffn.value.weight         f32 privateuseone:0   8192  2048
............................................................................................................................................................................................................................................................................................................................................................................................................
blocks.23.ln1.weight              f32 privateuseone:0   2048
blocks.23.ln1.bias                f32 privateuseone:0   2048
blocks.23.ln2.weight              f32 privateuseone:0   2048
blocks.23.ln2.bias                f32 privateuseone:0   2048
blocks.23.att.time_decay          f32 privateuseone:0   2048
blocks.23.att.time_first          f32 privateuseone:0   2048
blocks.23.att.time_mix_k          f32 privateuseone:0   2048
blocks.23.att.time_mix_v          f32 privateuseone:0   2048
blocks.23.att.time_mix_r          f32 privateuseone:0   2048
blocks.23.att.key.weight          f32 privateuseone:0   2048  2048
blocks.23.att.value.weight        f32 privateuseone:0   2048  2048
blocks.23.att.receptance.weight   f32 privateuseone:0   2048  2048
blocks.23.att.output.weight       f32 privateuseone:0   2048  2048
blocks.23.ffn.time_mix_k          f32 privateuseone:0   2048
blocks.23.ffn.time_mix_r          f32 privateuseone:0   2048
blocks.23.ffn.key.weight          f32 privateuseone:0   2048  8192
blocks.23.ffn.receptance.weight   f32 privateuseone:0   2048  2048
blocks.23.ffn.value.weight        f32 privateuseone:0   8192  2048
ln_out.weight                     f32 privateuseone:0   2048
ln_out.bias                       f32 privateuseone:0   2048
head.weight                       f32 privateuseone:0   2048 50277

Run prompt...
指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\n代表换行，必须用 Raven 模型
+ --> 让机器人换个回答
+reset --> 重置对话，请经常使用 +reset 重置机器人记忆

+i 某某指令 --> 问独立的问题（忽略聊天上下文），用\n代表换行，必须用 Raven 模型
+gen 某某内容 --> 续写内容（忽略聊天上下文），用\n代表换行，写小说用 testNovel 模型
+++ --> 继续 +gen / +i 的回答
++ --> 换个 +gen / +i 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

中文网文 testNovel 模型，可以试这些续写例子（不适合 Raven 模型！）：
+gen “区区
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\n第一章
+gen 这是一个修真世界，详细世界设定如下：\n1.

Chinese - ./fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-done-ctx2048-20230225 - privateuse1 fp32

The following is a coherent verbose detailed conversation between a Chinese girl named Alice and her friend Bob. Alice is very intelligent, creative and friendly. Alice likes to tell Bob a lot about herself and her opinions. Alice usually gives Bob kind, helpful and informative advices.

Bob: lhc
Alice: LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地 下建造。LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色 子的存在。

Bob: 企鹅会飞吗
Alice: 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行 。

Bob: hi
Alice:E:\Github\ChatRWKV-DirectML\v2/../rwkv_pip_package/src\rwkv\utils.py:78: UserWarning: The operator 'aten::multinomial' is not currently supported on the DML backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at D:\a\_work\1\s\pytorch-directml-plugin\torch_directml\csrc\dml\dml_cpu_fallback.cpp:17.)
  out = torch.multinomial(probs, num_samples=1)[0]
 你好，我是亚马逊的技术负责人。你好，我也是亚马逊的技术负责人。

Bob:
```

##  Q&A

### Q:I got a warning message, just like below, how about it?
```
E:\Github\ChatRWKV-DirectML\v2/../rwkv_pip_package/src\rwkv\utils.py:78: UserWarning: The operator 'aten::multinomial' is not currently supported on the DML backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at D:\a\_work\1\s\pytorch-directml-plugin\torch_directml\csrc\dml\dml_cpu_fallback.cpp:17.)
  out = torch.multinomial(probs, num_samples=1)[0]
```
### A: It is ok, It will use CPU to deal with the operator 'aten::multinomial', you can ignore it.


## Tutorial

[ChatRWKV｜开源中文小说以及文章生成语言模型](https://openai.wiki/chatrwkv.html)

---

> ChatRWKV README

---

# ChatRWKV (pronounced as "RwaKuv", from 4 major params: R W K V)
ChatRWKV is like ChatGPT but powered by my RWKV (100% RNN) language model, which is the only RNN (as of now) that can match transformers in quality and scaling, while being faster and saves VRAM. Training sponsored by Stability EleutherAI :) **中文使用教程，请往下看，在本页面底部。**

**HuggingFace Gradio Demo (14B ctx8192)**: https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio

**Raven** (7B finetuned on Alpaca and more) Demo: https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B

**RWKV pip package**: https://pypi.org/project/rwkv/ **(please always check for latest version and upgrade)**

Update ChatRWKV v2 & pip rwkv package (0.7.3):

Use v2/convert_model.py to convert a model for a strategy, for faster loading & saves CPU RAM.
```
### Note RWKV_CUDA_ON will build a CUDA kernel ("pip install ninja" first).
### How to build in Linux: set these and run v2/chat.py
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
### How to build in win:
Install VS2022 build tools (https://aka.ms/vs/17/release/vs_BuildTools.exe select Desktop C++). Reinstall CUDA 11.7 (install VC++ extensions). Run v2/chat.py in "x64 native tools command prompt". 
```

**Download RWKV-4 weights:** https://huggingface.co/BlinkDL (**Use RWKV-4 models**. DO NOT use RWKV-4a and RWKV-4b models.)

![ChatRWKV-strategy](ChatRWKV-strategy.png)

## RWKV Discord: https://discord.gg/bDSBUMeFpc (let's build together)

**Twitter:** https://twitter.com/BlinkDL_AI

**RWKV LM:** https://github.com/BlinkDL/RWKV-LM (explanation, fine-tuning, training, etc.)

**RWKV in 150 lines** (model, inference, text generation): https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py

ChatRWKV v2: with "stream" and "split" strategies, and INT8. 3G VRAM is enough to run RWKV 14B :) https://github.com/BlinkDL/ChatRWKV/tree/main/v2
```python
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # if '1' then use CUDA kernel for seq mode (much faster)
from rwkv.model import RWKV                         # pip install rwkv
model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040', strategy='cuda fp16')

out, state = model.forward([187, 510, 1563, 310, 247], None)   # use 20B_tokenizer.json
print(out.detach().cpu().numpy())                   # get logits
out, state = model.forward([187, 510], None)
out, state = model.forward([1563], state)           # RNN has state (use deepcopy if you want to clone it)
out, state = model.forward([310, 247], state)
print(out.detach().cpu().numpy())                   # same result as above
```
![RWKV-eval](RWKV-eval.png)

Here is https://huggingface.co/BlinkDL/rwkv-4-raven/blob/main/RWKV-4-Raven-14B-v7-Eng-20230404-ctx4096.pth in action:
![ChatRWKV](ChatRWKV.png)

Cool Community RWKV Projects:

https://pypi.org/project/rwkvstic/ pip package (with 8bit & offload for low VRAM GPUs)

**https://github.com/saharNooby/rwkv.cpp rwkv.cpp for fast CPU reference**

https://github.com/wfox4/WebChatRWKVv2 WebUI

https://github.com/cryscan/eloise RWKV QQ bot

The lastest "Raven"-series Alpaca-style-tuned RWKV 14B & 7B models are very good (almost ChatGPT-like, good at multiround chat too). Download: https://huggingface.co/BlinkDL/rwkv-4-raven

Previous old model results:
![ChatRWKV](misc/sample-1.png)
![ChatRWKV](misc/sample-2.png)
![ChatRWKV](misc/sample-3.png)
![ChatRWKV](misc/sample-4.png)
![ChatRWKV](misc/sample-5.png)
![ChatRWKV](misc/sample-6.png)
![ChatRWKV](misc/sample-7.png)

## 中文模型

QQ群 553456870（加入时请简单自我介绍）。有研发能力的朋友加群 325154699。

中文使用教程：https://zhuanlan.zhihu.com/p/618011122

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BlinkDL/ChatRWKV&type=Date)](https://star-history.com/#BlinkDL/ChatRWKV&Date)
