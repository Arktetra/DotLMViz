<script lang="ts">
	import { onMount } from "svelte";
	import { data } from "../state.svelte";
	import { curveBasis, extent, line, scaleLinear } from "d3";

    let { pre, post } : {
        pre: [number, number][],
        post: [number, number][]
    } = $props();

    let width = $state(500),
        height = $state(266),
        margin = { left: 20, right: 20, top: 20, bottom: 20 };

    let innerWidth = $derived(width - (margin.left + margin.right)),
        innerHeight = $derived(height - (margin.top + margin.bottom));

    let xPreMinMax = extent(pre, (d) => d[0]) as [number, number],
        yPreMinMax = extent(pre, (d) => d[1]) as [number, number],
        xPostMinMax = extent(post, (d) => d[0]) as [number, number],
        yPostMinMax = extent(post, (d) => d[1]) as [number, number];

    let xPreMin = xPreMinMax[0];

    let xMin = Math.min(xPreMinMax[0], xPostMinMax[0]),
        xMax = Math.max(xPreMinMax[1], xPostMinMax[1]),
        yMin = Math.min(yPreMinMax[0], yPostMinMax[0]),
        yMax = Math.max(yPreMinMax[1], yPostMinMax[1]);

    let xScale = $derived(
        width && pre && post
        ? scaleLinear()
            .domain([xMin, xMax])
            .range([margin.left, width - margin.right])
        : null
    );

    let yScale = $derived(
        height && pre && post
        ? scaleLinear()
            .domain([yMin, yMax])
            .range([height - margin.bottom, margin.top])
        : null
    );

    let lineGenerator = $derived(
        xScale && yScale
        ? line()
            .curve(curveBasis)
            .x((d) => xScale(d[0]))
            .y((d) => yScale(d[1]))
        : null
    );

    onMount(() => {
        const chart = document.querySelector(".chart")

        if (chart) {
            height = chart.clientHeight;
        }
    })
</script>

<div class="chart" bind:clientWidth={width}>
    {#if pre && post && xScale && yScale && lineGenerator}
        <svg {width} {height}>
            <path
                d={lineGenerator(pre)}
                fill="#fcd34d"
                opacity="0.8"
                stroke="#000"
                stroke-width="1"
                stroke-linejoin="round"
            />
            <path
                d={lineGenerator(post)}
                fill="#f0ffff"
                opacity="0.8"
                stroke="#000"
                stroke-width="1"
                stroke-linejoin="round"
            />
        </svg>
    {/if}
</div>

<style>
    .chart {
        height: 100%;
    }
</style>