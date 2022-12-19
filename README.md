An image processing program that can sharpen or remove noise from a given message

Requires Nvdia's Warp API
```text
    pip3 install warp
```

Instructions on running the program
```text
Your executable will run as follows:
python3 imageProcessing.py algType kernSize param inFileName outFileName
where:
- algType is either -s (sharpen) or -n (noise removal)
- kernSize is the kernel size - e.g. 3 for 3x3, 5 for 5x5, etc.. It must always be positive and
odd.
- param is the additional numerical parameter that the algorithm needs - e.g. the scaling value
k for unsharp masking or sigma for the gaussian. If your algorithm doesn't need any additional parameters once it knows the kernel size, just pass some dummy value here (e.g. 0)
- inFileName is the name of the input image file
- outFileName is the name of the output image file
```