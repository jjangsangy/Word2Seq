# Training a Recurrent Neural Network for Text Generation

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)

![harry potter](https://github.com/jjangsangy/Word2Seq/raw/master/static/harry.gif)

This implements a char-rnn, as made famous by
[Andrei Karpathy's work](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) on
text generation using recurrent neural networks.

# Training on GPU vs CPU
It is recommended to run this script on GPU, as recurrent networks are quite computationally intensive. If access to GPU optimized tensorflow is not available, it's sometimes possible to find precompiled binaries on Github.

I have compiled some optimized [tensorflow binaries](https://github.com/jjangsangy/MacOS-TensorflowBuilds) for `MacOS` that people can use running the latest tensorflow and utilizing Intel Math Kernel Library.

# Corpus Sizes
Make sure your corpus has >100k characters at least, but for the best >1M characters. Relatively, that is around the size, of Harry Potter Book 7.

# Architecture

We implement a Stateful Stacked LSTM in Keras that can be customized on the command line.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 40, 128)           110080
_________________________________________________________________
lstm_2 (LSTM)                (None, 40, 128)           131584
_________________________________________________________________
lstm_3 (LSTM)                (None, 128)               131584
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
output (Dense)               (None, 86)                11094
_________________________________________________________________
activation_1 (Activation)    (None, 86)                0
```

# Examples

I've distributed the binary weights and models used to generate these segments which can be found in the `models` directory.

## The Illiad of Homer

```
BOOK II.


Heckees, Sestos and Adresteia draw as he              100
  Horden arms by this beast guard killed.
  Ark, though batch offered the cues.
  Is mine enchanted his wrist kingly watch
  Threatened benefit of his knee is these.
  Caring wet table to be bespoke herself              420
  For battle, damed us them spoken I deemed.
  Nor purpiring in neck he well-armown,
  Ilium I rathor! Dolyage the cleaped Daon
  The yare tiblevelin, the midef glow
  of-minded swarming reptoul leaned
  Shall foughter for his, nor hed oft which
  Onforthirul dead may boatiant sost                   570
Of Meris, at will to epthe wavecus,
  And give should were weal cup obloed, arms,
  But in their dowert worb his ruch tongs
  Thy deuphted of his I led hair bedeme,
  Ademorodes' tremble ladnerss thee, to Aiche.
  My both sindal ordreel beard she willfully
  In dung'd tryinigable home Meanely.
  Senoth if blare righter of warring aid.             290
  Hastes his fortancling sarius call'd ittens
  Poly phe so ace nelty twy agirs on deepsife         225
```

## Harry Potter and The Deathly Hallows

```
“I say we jump when it gets low enough!”

And he realized, they were leaving along and began to
ask over the star and where the green nan
langed to the barman knocker on the bed and heard.

But Ron, and Hermione was pointing at Harry
who had bleeded. He really had to expect.

It had gotted to move again, but under the thought
we had to, Harry with an elf was done. It was followed
that the goblins striding facing them.

“Are in, it’s not toast the thing” said Malfoys into
the lost that the stone was now is younger.

“Wonks, will it, in here!” How not did the school
sitting like words to his face and thought?

“You are taken” said Ron.

How long for the tiny black of his actual and then held
on the long while the thing else in it.

Had perswaind through the side of his night floor.

Harry went for this bleeding of Dobby. Snape had
prefered him, as though swelled initially.
He asked, and shouldered

“Gurblevy blood morned and she would have been two
of the officious for the corner”

“Oh, the interested when you worked with something that
west?”

“What was young in, you can’t be up hard, on the Dark
Lord turned,” and Harry excited them, leaving as
the goat of her back again. The sounds of
the two, and uncentantable with nestprotures of preuch
around to a thickness for the blood and rather into the
goblin and took room.

Harry and thought it was claying the search of
the darkness from his head to the protects and
entered in the few sunation of she called in the
door — because  Dumbledore had made when
the table of side of the distance of them snow,
and you can have it was firsts
and shouted  "Disappeararall!".

I was a while what was mletograls.

It laughed partly.

“Mad.”
```

## The Linux Kernel

```c
/*
 * Copyright (C) 2006 Bassififo 11RP
 * subplied the invert the II detach this of the Linux numin this himes.
 * DID subloge in the packet processor permitation
 *
 * This program is distributed in the hope that it will be useful
 * but WITHOUT ANY WARRANTY
 *
 * without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHTTHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#include "stmbions.h"
#include "msc.h"
#include <linux/io.h>
#include <linux/export.h>
#include <linux/init.h>

static struct *rc_map_ligo dumar_memcache = {
	.name	= "C0c424",
	.chip 	= {
		.init	= simmar->dummave,
		.flags	= CORT_SCAP_SCUEC,
		.base.oclass	= RE,
		.to_parent	= nv50_disp_ops,
		.remove		= 0;
	};
};

/*
 * Device drivers and TIL) open initialize
 * or the run be and cases volatile.
 */
static int __exit_exit_mpc_module(struct pcm_device *dev,
				  struct of_device_id *pdev,
				  struct hw_config *parent)
{
	struct cck_inode *proto;
	struct platform_device *pdev;
	const struct clk_ops *bus_bit;
	struct clk_regmap *register;

	if (IS__ONFO_LSPAD) {
		return -ENODEV;
	}
	if (!(parent < 0)) {
		memlport(sizeof(struct sk_muachice *));
		return -ENODEV;
	}
	if !(ret) {
		return idxi_register_phy(PC_PROMST_ORE);
	}
}

static int __exit_exit(struct sl_bus_func *ebt_reg, struct nvkm_map *dev)
{
	struct clk_ops *rmutex;
	size = cpu_lock;
	if (!data) {
		return;
	}
}
```

## Jay Chou Lyrics

> Lyrics data and implementation originated from this [blog post](http://leix.me/2016/11/28/tensorflow-lyrics-generation/) and [implementation](https://github.com/leido/char-rnn-cn) by [leido](https://github.com/leido)
```
作詞：周杰倫
作曲：方文倫

給你已經很久

我們走了 我沒有愛
你說我還是不能承諾
你說把一種龍 都回遠
選才是誰說你的愛溢出就象雨水

邪火等待我們的周杰倫
皇室的總決想一種解藥
我們微笑那傷
鬧人們 你確定了那角向

我不能再想
我不能再想

我不是再多 給不該一口氣
隨后中一百悔在風天邊
朝著起一百悔在假牌洗刷

你卻給你的愛情　
你說我不該沒有
你不知道 不要我
我不想要我
我不 我不要再想

我知道說更苦
我們在秘密
我們給你的臉

還讓我們追求 不想要走
細數慚愧我們都不到
我知道不能
你情底還想我和唱再想

就過的電方裡
我們黯一頁會名

你的多小到聽會痛走
細數慚愧我傷你幾回
我灌溉 一果
狼懼月 東方就剛可
簡單綠白卻又再考倒 我
說散你想很久了吧?

卻等拳跟離別
長漢裡 未大下量
鳥飛翔開的地方
我等幾個世界

越來一起旅行
北在窗盤的床
周圍的眼眸
跟藥跳裡 幸持寫的空量
我輕輕地嚐一口 份量雖然不多

我不能我彷彿都這麼
脆弱 籬蜓說的感覺
我們擁以一起的山跳進
我們的愛 (不能)

爸非是我們乘著陽光
我車法詩離開
正表情對遠方的把叫一天
我們在一隻貓
動作輕盈地圍 卻燒不了也有一疊

渴望們她遠死了北
叛軍如傲 無名風吹看著日
你終甲 我們帶著你的愛寫了
我的世界 你在眼神看著我
我的身影
我的感覺
我們的愛完在
你那法擊嗇
而我給 我的選擇
它在身影
青塵埃
```

## Fox News

```
Fox News
Date: 2016-11-07
Trump and the Bond

“Trump is, as the district," said State Department’s new strategist.
The legal officials people and it would be the new president,
every "entiple winning to this dasham other appeared many and
policy, and is new strategist”.

For President America. By the latest running
presidential Hillary Clinton isn’t a christian. The issue of
internet in on email and friends.

“To don't continue to be to child” likely to be not be one
Clinton told Fox.

The 2015 on Hillary Clinton was released the she get to go or
nearly campaign even the president for the finalist to top
about the Washington and a bird resting possible to
be removed latest between the forward.

The source and subtracting to do a case of exiles of the first state
More of a more decision and secretary into the battering the FBI and
A Cheney Republican said "Trump is criminal server, who went”
Portion to be this were voters and any team would have this criticism
who this company do several diplomatic history emails with Trump on a state's endorse.
```

## CoinDesk

```
According to The DAO

According to The DAO official financial services and compliance ledger finances are taken with the market of a distributed ledger statement:

"It's the company's international industry and previously will be published you decided what can see a compare the exchange and founders.

Additional entry plans to be wide revealed to include call the detail of the affective
in effectively data, one open solutions of two wider way to a revalue more ensure the US
as the ability to the government with the smart contract funding with a major
month and fell which has adopted a new, there are described a
hacking currently raising a "Option (was overall in the data.

And are looking to trick the volume on the Ethereum Foundation,
has made that two questions, means information,
but it will make any character operates at what is statented with
coverage was a realize management in Assam Paar used on on the most ways to start.
```

# Installation

Currently only runs on Python 3 (because I can), you can install dependencies using `pip`

```
$ pip3 install -r requirements.txt
$ 
```

## Training

Place your corpuse[s] into the `datasets` folder and run

```sh
$ python3 train.py
```

To customize the parameters for generating text you can parameterize with input arguments.

```sh
$ python3 train.py --help

usage: train.py [-h] [--verbose] [--batch size] [--model file]
                [--window length] [--log_dir directory] [--layers deep]
                [--dropout amount] [--skip amount]

Train a neural network

optional arguments:
  -h, --help            show this help message and exit
  --verbose, -v         Keras verbose output
  --batch size, -b size
                        Specify the input batch size
  --model file, -m file
                        Specify the output model hdf5 file to save to:
                        [default]: models/model.h5
  --window length, -w length
                        Specify the size of the window size to train on:
                        [default]: 40
  --log_dir directory, -r directory
                        Specify the output directory for tensorflow logs:
                        [default]: None
  --layers deep, -l deep
                        Specify the number of layers deep of LSTM nodes:
                        [default]: 3
  --dropout amount, -d amount
                        Amount of LSTM dropout to apply between 0.0 - 1.0:
                        [default]: 0.2
  --skip amount, -s amount
                        Amount to skip within the dataset, integer from 1:
                        [default]: 1
```

## Text Generation

```sh
$ python3 decoder.py
```

## Debugging

To debug we've written log files in the log directory. In order to access these logs, you can run tensorboard.

```sh
$ tensorboard --logdir=./logs
```

![graph](https://github.com/jjangsangy/Word2Seq/raw/master/static/graph.png)
![tensorboard](https://github.com/jjangsangy/Word2Seq/raw/master/static/tensorboard.png)