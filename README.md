An image processing program that can sharpen or remove noise from a given message

Requires Nvdia's Warp API
```text
    pip3 install warp
```

Instructions on running the program
```text
    python3 imageProcessing.py algType kernSize param inFileName outFileName
    where:
    - algType is either -s (sharpen) or -n (noise removal)
    - kernSize is the kernel size - e.g. 3 for 3x3, 5 for 5x5, etc.. It must always be positive and
    odd.
    - param is the additional numerical parameter that the algorithm needs - e.g. the scaling value
    k for unsharp masking
    - inFileName is the name of the input image file
    - outFileName is the name of the output image file
```
