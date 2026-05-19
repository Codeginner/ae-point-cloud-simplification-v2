"""
plot_architecture.py
====================
Visualisasi arsitektur PointCloudSimplifier menggunakan PlotNeuralNet.

Setup:
    git clone https://github.com/HarisIqbal88/PlotNeuralNet
    cd PlotNeuralNet
    cp /path/to/plot_architecture.py pyexamples/
    bash tikzmake.sh plot_architecture

Output: pyexamples/plot_architecture.pdf
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *


def to_custom_block(name, s_filter, n_filter, offset, to, width, height, depth,
                    caption="", color="\\ConvColor"):
    return r"""
\pic[shift={""" + offset + r"""}] at (""" + to + r""") {
    Box={
        name=""" + name + r""",
        caption=""" + caption + r""",
        xlabel={{""" + str(s_filter) + r""", }},
        ylabel=""" + str(n_filter) + r""",
        zlabel=""" + str(n_filter) + r""",
        fill=""" + color + r""",
        height=""" + str(height) + r""",
        width=""" + str(width) + r""",
        depth=""" + str(depth) + r"""
    }
};
"""


arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # ── INPUT POINT CLOUD ──────────────────────────────────────────────────
    to_Conv(
        name="input",
        s_filer=1024,
        n_filer=3,
        offset="(0,0,0)",
        to="(0,0,0)",
        width=1,
        height=40,
        depth=40,
        caption="Input\\\\P (N,3)",
    ),

    # ── DGCNN ENCODER ──────────────────────────────────────────────────────
    # EdgeConv 1: 64
    to_Conv(
        name="ec1",
        s_filer=1024,
        n_filer=64,
        offset="(2,0,0)",
        to="(input-east)",
        width=2,
        height=40,
        depth=40,
        caption="EdgeConv\\\\64",
    ),
    to_connection("input", "ec1"),

    # EdgeConv 2: 64
    to_Conv(
        name="ec2",
        s_filer=1024,
        n_filer=64,
        offset="(1.5,0,0)",
        to="(ec1-east)",
        width=2,
        height=36,
        depth=36,
        caption="EdgeConv\\\\64",
    ),
    to_connection("ec1", "ec2"),

    # EdgeConv 3: 128
    to_Conv(
        name="ec3",
        s_filer=1024,
        n_filer=128,
        offset="(1.5,0,0)",
        to="(ec2-east)",
        width=3,
        height=32,
        depth=32,
        caption="EdgeConv\\\\128",
    ),
    to_connection("ec2", "ec3"),

    # EdgeConv 4: 256
    to_Conv(
        name="ec4",
        s_filer=1024,
        n_filer=256,
        offset="(1.5,0,0)",
        to="(ec3-east)",
        width=4,
        height=28,
        depth=28,
        caption="EdgeConv\\\\256",
    ),
    to_connection("ec3", "ec4"),

    # Concat → 448
    to_Conv(
        name="feat",
        s_filer=1024,
        n_filer=448,
        offset="(2,0,0)",
        to="(ec4-east)",
        width=6,
        height=40,
        depth=40,
        caption="f\\_i\\\\(N, 448)",
    ),
    to_connection("ec4", "feat"),

    # ── NC SCORE MODULE (above, teal) ──────────────────────────────────────
    to_Conv(
        name="nc",
        s_filer=1024,
        n_filer=1,
        offset="(0,7,0)",
        to="(feat-east)",
        width=1,
        height=40,
        depth=8,
        caption="NC Score\\\\s\\_i (N,1)",
    ),

    # ── IMPORTANCE SCORER ──────────────────────────────────────────────────
    to_Conv(
        name="scorer",
        s_filer=1024,
        n_filer=1,
        offset="(3,0,0)",
        to="(feat-east)",
        width=1,
        height=40,
        depth=8,
        caption="Scorer\\\\MLP\\\\score(N)",
    ),
    to_connection("feat", "scorer"),

    # ── ADAPTIVE SELECTOR ──────────────────────────────────────────────────
    to_SoftMax(
        name="selector",
        s_filter=512,
        offset="(2,0,0)",
        to="(scorer-east)",
        width=1,
        height=25,
        depth=25,
        caption="Adaptive\\\\Selector\\\\(STE)",
    ),
    to_connection("scorer", "selector"),

    # ── SIMPLIFIED POINTS P_s ──────────────────────────────────────────────
    to_Conv(
        name="ps",
        s_filer=512,
        n_filer=3,
        offset="(2,0,0)",
        to="(selector-east)",
        width=1,
        height=25,
        depth=25,
        caption="P\\_s\\\\(M, 3)",
    ),
    to_connection("selector", "ps"),

    # ── SIMPLIFIED FEATURES f_s ────────────────────────────────────────────
    to_Conv(
        name="fs",
        s_filer=512,
        n_filer=448,
        offset="(0,0,0)",
        to="(ps-east)",
        width=6,
        height=25,
        depth=25,
        caption="f\\_s\\\\(M, 448)",
    ),

    # ── FOLDINGNET DECODER (below) ─────────────────────────────────────────
    # Global MLP
    to_Conv(
        name="gmlp",
        s_filer=512,
        n_filer=1024,
        offset="(0,-8,0)",
        to="(fs-south)",
        width=8,
        height=10,
        depth=25,
        caption="Global MLP\\\\z (1024)",
    ),
    to_connection("fs", "gmlp"),

    # Fold stage 1
    to_Conv(
        name="fold1",
        s_filer=512,
        n_filer=3,
        offset="(2,0,0)",
        to="(gmlp-east)",
        width=1,
        height=10,
        depth=25,
        caption="Fold 1\\\\(M,3)",
    ),
    to_connection("gmlp", "fold1"),

    # Fold stage 2 → P_recon
    to_Conv(
        name="precon",
        s_filer=512,
        n_filer=3,
        offset="(2,0,0)",
        to="(fold1-east)",
        width=1,
        height=10,
        depth=25,
        caption="P\\_recon\\\\(M, 3)",
    ),
    to_connection("fold1", "precon"),

    # ── CLS HEAD (above fs) ────────────────────────────────────────────────
    to_Conv(
        name="pool",
        s_filer=1,
        n_filer=896,
        offset="(0,6,0)",
        to="(fs-north)",
        width=6,
        height=6,
        depth=25,
        caption="Max+Mean\\\\pool (896)",
    ),
    to_connection("fs", "pool"),

    to_SoftMax(
        name="logits",
        s_filter=10,
        offset="(2,0,0)",
        to="(pool-east)",
        width=1,
        height=6,
        depth=6,
        caption="Logits\\\\(num\\_class)",
    ),
    to_connection("pool", "logits"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
