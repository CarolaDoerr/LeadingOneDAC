# Experiments for Section 3

This document explains how to generate the data depicted in Section 3 and where to find it.
This covers the data for

* Tables 1 and 2 and
* Figures 1 and 2.

When we refer to `root` (also written as `.`) in this document, we mean the folder that contains this `README.md`.
Further, we assume a Windows operating system.
Thus, directories are separated via `\`, and executable files end in `.exe`.

## Compile and Run the Code

The code is written in the [D programming language](https://dlang.org/).
We recommend to use the compiler [LDC](https://github.com/ldc-developers/ldc#installation), as it has a high performance, but the [official D compiler (DMD)](https://dlang.org/download.html#dmd) works too.
The following instructions assume that LDC is installed and that the compiler (`ldc2.exe`) is found in the shell.

In order to compile the code, run the following command from `root` in a shell:

```shell
> ldc2 -release .\source\app.d .\source\data_management.d .\source\eas.d
```

This produces (besides others) the file `app.exe` in `root`. Run this file in a shell via:

```shell
> .\app.exe
```

Running `app.exe` might take some time, as the computation of the optimal portfolio uses a brute-force approach.
However, during the execution, new directories should be produced in `root`.

## Find the Data

This section assumes that the experiments were ran successfully, as described in [the previous section](README.md#compile-and-run-the-code).
It explains where to find the data that is depicted in Section 3 of the paper.

### Tables 1 and 2

The data for the tables is scattered among the different folders produced by `app.exe`.
For Table 1, for a specific $k$ and a specific $n$, the respective data is found in `.\information\k=`$k$`\n=`$n$`_optimal_portfolio`.
The portfolio itself is in the first line of the file, the relative optimal expected runtime in line 5.

For Table 2, the data is spread among the same files as for Table 1, with the difference that the file name is not necessarily `n=`$n$`_optimal_portfolio` but the part `optimal_portfolio` is replaced by the tag of the portfolio from the paper (with the exception that `optimal` from the paper is called `optimal_portfolio` here).

## Figures 1 and 2

The plots are reproduced by compiling the TeX files in subdirectories of `.\visualization\plots`.
For Figure 1, compile `.\visualization\plots\cumulative_optimal_policies\cumulative_plot_n=50_k=3.tex`.
For Figure 2, compile `.\visualization\plots\vary_k\optimal\optimal_vary_k_n=50.tex`.

We note that the orange labels in Figure 1 are added manually to the plot.
In order to get the correct data for the experiments, please use the data for the relative optimal expected runtime as stated [above for Tables 1 and 2](README.md#tables-1-and-2).
