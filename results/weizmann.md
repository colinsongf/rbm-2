### WEIZMANN HORSE

resized to 28\*28 / 32\*32

- reproduce shapebm

>  pure FC: 500-1000-1000

Both resulted in all outputs being the same: horse without legs  ![epoch4-plot](weizmann/horse_fc_epoch4-plot.png)

>  Conv to simulate FC: (16,16,1,500) with stride (14,14)

![shapebm_epoch49-plot](weizmann/shapebm_epoch49-plot.png)

- pure conv without FC on 32\*32

> (4, 4, 1, ?), (4, 4) # 8\*8
>
> (4, 4, ?, ?), (4, 4) # 2\*2
>
> (2, 2, ?, ?), (2, 2) # 1\*1

pixels from the same receptive fields seem to have unified value

![epoch12-plot](weizmann/pure_conv_without_fc_epoch12-plot.png)

- pure conv (same stride as filter size ![) with FC on 32\*32

> (4, 4, 1, ?), (4, 4) # 8\*8
>
> (4, 4, ?, ?), (4, 4) # 2\*2
>
> FC xxx

noise, but this seem to have bad but still recognizable result on mnist

 ![same_stride_with_fc_epoch0-plot](weizmann/same_stride_with_fc_epoch0-plot.png)