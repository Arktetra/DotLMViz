<script lang="ts">
    import { scaleLinear } from "d3-scale";
    import Circle from "./Circle.svelte";
	import { onMount } from "svelte";
	import { extent } from "d3";
	import Quadtree from "./Quadtree.svelte";

    let { data } : {
        data: ScatterPlotData
    } = $props();

    let width = $state(500);
    let height = $state(266);
    const padding = { top: 20, right: 15, bottom: 20, left: 20};

    let xScale = $derived(scaleLinear()
        .domain(extent(data, (d) => d.x) as [number, number])
        .range([padding.left, width - padding.right])
    );

    let yScale = $derived(scaleLinear()
        .domain(extent(data, (d) => d.y) as [number, number])
        .range([height - padding.bottom, padding.top])
    );

    onMount(() => {
        const chart = document.querySelector(".chart");

        if (chart) {
            height = chart.clientHeight;
        }
    });
</script>

<div class="chart" bind:clientWidth={width}>
    <div class="relative">
        <Quadtree
            {xScale}
            {yScale}
            margin={padding}
            {data}
            searchRadius={30}
        >
        {#snippet children({x, y, found, visible})}
            <div
                class="circle"
                style="
                    top:{y}px;
                    left:{x.circle}px;
                    display: {visible
                        ? 'block'
                        : 'none'
                    };
                    width: {10};
                    height: {10};
                ">
            </div>
            <div
                class="tooltip"
                style="
                    top:{y! + 5}px;
                    left: {x.square + 10}px;
                    display: {visible
                        ? 'block'
                        : 'none'
                    };
                "
            >
                <h1 class="tooltip-heading">{"token" in found ? found.token : ""}</h1>
                x: {(found.x).toFixed(3)}<br />
                y: {(found.y).toFixed(3)}
            </div>
        {/snippet}
        </Quadtree>

        <svg {width} {height}>
            <g class="bars">
                {#each data as point}
                    <Circle
                        x={xScale(point.x)}
                        y={yScale(point.y)}
                        r={5}
                        fill={"#6a6ef0"}
                    />
                {/each}
            </g>
        </svg>
    </div>
</div>

<style>
    .chart {
        height: 100%;
        position: relative;
    }

    svg {
        background-color: aliceblue;
        color: #6a6ef0;
    }

    .circle {
        position: absolute;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        width: 10px;
        height: 10px;
        border: 1px solid #000000;
        transition:
            left 300ms ease,
            top 300ms ease;
    }

    .tooltip {
        position: absolute;
        font-family: 'Poppins', sans-serif !important;
        min-width: 8em;
        line-height: 1.2;
        pointer-events: none;
        font-size: 0.875rem;
        z-index: 1;
        padding: 6px;
        background-color: #f0f0f080;
        color: #555555;
        transition:
            left 100ms ease,
            top 100ms ease;
        /* fill-opacity: 0.5; */
        text-align: left;
    }

    .tooltip-heading {
        color: black;
    }
</style>