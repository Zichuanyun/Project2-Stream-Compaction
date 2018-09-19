CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Zichuan Yu
  * [LinkedIn](https://www.linkedin.com/in/zichuan-yu/), [Behance](https://www.behance.net/zainyu717ebcc)
* Tested on: Windows 10.0.17134 Build 17134, i7-4710 @ 2.50GHz 16GB, GTX 980m 4096MB GDDR5

## Features

- CPU Scan
- CPU Stream Compaction
- Naive GPU Scan
- Work-Efficient GPU Scan
- Work-Efficient GPU Stream Compaction
- Thrust Implementation

## Performance Analysis

### Block size analysis

![block_size](img/block_size.png)

### Array Size Analysis on Scan

![scan](img/scan.png)

### Array Size Analysis on Compaction

![compaction](img/compaction.png)

## Output

Array size 2^26, block size 1024

```shell

****************
** SCAN TESTS **
****************
    [   1   1   1   1   1   1   1   1   1   1   1   1   1 ...   1   1 ]
==== cpu scan, power-of-two ====
   elapsed time: 1535.85ms    (std::chrono Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 268435454 268435455 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 594.798ms    (std::chrono Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 268435451 268435452 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 510.046ms    (CUDA Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 268435454 268435455 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 510.037ms    (CUDA Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 175.304ms    (CUDA Measured)
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 268435454 268435455 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 175.151ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 28.8416ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 28.8394ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   0   0   1   0   3   1   3   3   0   3   1 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 708.621ms    (std::chrono Measured)
    [   1   1   3   1   3   3   3   1   1   1   1   1   1 ...   1   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 680.761ms    (std::chrono Measured)
    [   1   1   3   1   3   3   3   1   1   1   1   1   1 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1471.92ms    (std::chrono Measured)
    [   1   1   3   1   3   3   3   1   1   1   1   1   1 ...   1   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 213.044ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 212.931ms    (CUDA Measured)
    passed
Press any key to continue . . .
```






