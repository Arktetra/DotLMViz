<script lang="ts">
    import { scaleBand, scaleLinear } from "d3-scale";
	import { LabelSolid } from "flowbite-svelte-icons";
	import { onMount } from "svelte";
	import { global_state } from "../state.svelte";

    let {
        data,
        vmin="#e00000",
        vmax="#0000e0",
        xlabel="source token",
        ylabel="destination token",
    } = $props<{data: Array<Array<number>>}>();

    let minRow = data.map((row: Array<number>) => Math.min(...row));
    let maxRow = data.map((row: Array<number>) => Math.max(...row));
    let min = Math.min(...minRow);
    let max = Math.max(...maxRow);

    let width = $state(500);
    let height = $state(266);

    let padding = {
        top: 20,
        right: 15,
        bottom: 30,
        left: 30
    }

    let xScale = $derived(scaleLinear()
        .domain([0, data.length])
        .range([padding.left, width - padding.right])
    );

    let yScale = $derived(scaleLinear()
        .domain([0, data.length])
        .range([height - padding.bottom, padding.top])
    );

    let blueScale = scaleLinear<string>()
        .domain([0, max])
        .range(["white", vmax]);

    let redScale = scaleLinear<string>()
        .domain([min, 0])
        .range([vmin, "white"])

    function RGB(val: number) {
        if (val < 0) {
            return redScale(val);
        } else {
            return blueScale(val);
        }
    }

    let innerHeight = $derived(height - (padding.top + padding.bottom));
    let innerWidth = $derived(width - (padding.left + padding.right));

    let box_width = $derived(innerWidth / data.length);
    let box_height = $derived(innerHeight / data.length);

    onMount(() => {
        const chart = document.querySelector(".chart")

        if (chart) {
            height = chart.clientHeight;
        }
    });

    $effect(() => {
        console.log(height);
        console.log(padding);
        console.log("top:" + yScale(0));
        console.log("bottom:" + yScale(data.length));
    })
</script>

<div class="chart" bind:clientWidth={width}>
    <svg {width} {height}>
        {#each data as row, i}
            {#each row as val, j}
                <rect
                    x={xScale(j)}
                    y={yScale(data.length - i)}
                    width={box_width}
                    height={box_height}
                    fill={`${RGB(val)}`}
                    stroke="blue"
                    stroke-width=1
                />
            {/each}
        {/each}

        <!-- Y label -->
        <text
            x={xScale(0)}
            y={yScale(data.length / 2)}
            text-anchor="middle"
            transform="rotate(-90, {xScale(0)- 10}, {yScale(data.length / 2)})"
        >{ylabel}</text>

        <!-- X label -->
        <text
            x={xScale(data.length / 2)}
            y={yScale(0)}
            dy=15
            text-anchor="middle"
            transform="rotate(0, {xScale(0)}, {yScale(data.length / 2)})"
        >{xlabel}</text>
    </svg>
</div>

<style>
    .chart {
        height: 100%;
    }

    svg {
        background-color: aliceblue;
        color: #6a6ef0;
    }
</style>