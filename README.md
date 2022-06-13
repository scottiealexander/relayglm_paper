# relayglm_paper

Code to recreate figures from [Dynamics of temporal integration in the lateral geniculate nucleus](https://github.com/scottiealexander/relayglm_paper.git).

## Usage

* This code requires [Julia](https://julialang.org/), if you do not already have Julia installed you can download it from [https://julialang.org/downloads/](https://julialang.org/downloads/).
    * **NOTE**: this code has only been tested on Julia version 1.6.1, versions between 1.2.x and 1.6.1 *should* all work, they are, however, untested.
* Download or clone code into a suitable location (we will refer to the path to the directory holding the code as `<code_dir>`)
* Launch `julia`:

```julia
code_dir = "..." #location of the cloned repo
cd(code_dir)
```

### Install dependencies

Run the following (you only need to do this ***ONCE***)

```julia
# install dependencies (you only need to do this once)
include("./install_dependencies.jl")

# use of local Pkg environment / project is recommended, pass false to install
# deps to the home project
install_dependencies(true)
```

### Setup the local Pkg environment
Run the following code each time you launch a new Julia session (and want to recreate the figures):

```Julia
code_dir = "..." #location of the cloned repo
cd(code_dir)

# make sure Julia can find all the figure and utility modules (needed once per Julia session)
include("./setpath.jl")

# ============================================================================ #
# recreate figure 1
using Figure1

# use Pair 214 as our example as the paper does
d1 = Figure1.collate_data(214)

# make the figure
Figure1.make_figure(d1, 214)

# ============================================================================ #
# recreate figure 2
using Figure2

d2 = Figure2.collate_data()

# use Pair 208 as the example (same as paper does)
Figure2.make_figure(d2, 208)

# ============================================================================ #
# recreate figure 3
using Figure3

d3 = Figure3.collate_data()

# use Pair 208 as the example (same as figure 2 and as paper does)
Figure3.make_figure(d3, 208)

# ============================================================================ #
# recreate figure 4
using Figure4

d4 = Figure4.collate_data()

Figure4.make_figure(d4)
# ============================================================================ #
# recreate figure 5
using Figure5

d5 = Figure5.collate_data()

Figure5.make_figure(d5)
# ============================================================================ #
# recreate figures 6 & 7
using Figure6_7

d67 = Figure6_7.collate_data()

Figure6_7.make_figure(d67)

# ============================================================================ #
# recreate figure 8
using Figure8

# we can specify the time window durations to use for RGC spike partitioning
# here we use 100 ms (or 0.1 sec) which is what the Figure6 in the paper uses
d8 = Figure8.collate_data(twin=0.1)

Figure8.make_figure(d8)

# ============================================================================ #
# recreate figure 9
using Figure9

# NOTE: the data for Figures 6 and 7 are the same, so if you just ran the
# analysis for figure 6 you can just use the following:
d9 = d8

# # otherwise...
# d9 = Figure9.collate_data()

Figure9.make_figure(d8)

# ============================================================================ #
# recreate figure S1
using FigureS1

# NOTE: the data for Figures 6\7 and S1 are the same, so if you just ran the
# analysis for figures 6\7 you can just use the following:
ds1 = d67

# otherwise...
ds2 = FigureS2.collate_data()

FigureS2.make_figure(ds2)

# ============================================================================ #
# recreate figure S2
using FigureS2

ds2 = FigureS2.collate_data()
FigureS2.make_figure(ds2)

# ============================================================================ #
# recreate figure S3
using FigureS3

# NOTE: in the paper GLM models are simulated for 50 iterations, but that takes
# quite some time, here we'll just use 5 to speed things up
ds3 = FigureS3.collate_data(niter=5)
FigureS3.make_figure(ds3)

# ============================================================================ #
# recreate figure S4
using FigureS4

ds4 = FigureS4.collate_data()
FigureS4.make_figure(ds4)

```

## Dependencies

The non-registered dependencies on which this code relies are automatically installed by the script `install_dependencies.jl` (see instructions above). Their code can be found at:

* [RelayGLM.jl](https://github.com/scottiealexander/RelayGLM.jl.git)
* [PairsDB.jl](https://github.com/scottiealexander/PairsDB.jl.git)
* [WeyandDB.jl](https://github.com/scottiealexander/WeyandDB.jl.git)

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
