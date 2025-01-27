<script lang="ts">
	import { extent } from "d3";
    import { scaleLinear } from "d3-scale";
	import { onMount } from "svelte";
	import { activeComponent, global_state } from "../state.svelte";

    let {
        data,
        vmin="#e00000",
        vmax="#0000e0",
        xlabel="source token",
        ylabel="destination token",
    } : {
        data: HeatMapData,
        vmin?: string,
        vmax?: string,
        xlabel?: string,
        ylabel?: string,
    } = $props();

    let minmax = extent(data, (d) => d.score) as [number, number];
    let min = minmax[0];
    let max = minmax[1];

    let width = $state(500);
    let height = $state(266);

    let found: HeatMapPoint | null = $state(null);
    let visible: boolean = $state(false);

    let padding = {
        top: 20,
        right: 15,
        bottom: 30,
        left: 30
    }

    let xScale = $derived(scaleLinear()
        .domain([0, global_state.tokens.length])
        .range([padding.left, width - padding.right])
    );

    let yScale = $derived(scaleLinear()
        .domain([0, global_state.tokens.length])
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

    let box_width = $derived(innerWidth / global_state.tokens.length);
    let box_height = $derived(innerHeight / global_state.tokens.length);

    onMount(() => {
        const chart = document.querySelector(".chart")

        if (chart) {
            height = chart.clientHeight;
        }
    });

    $effect(() => {
        // console.log(height);
        // console.log(padding);
        // console.log("top:" + yScale(0));
        // console.log("bottom:" + yScale(global_state.tokens.length));
        $inspect(data);
        $inspect(global_state.tokens.length);
    })
</script>


<div class="chart" bind:clientWidth={width}>
    <div class="relative">
        {#if found}
            <div
                class="rect"
                style="
                    top: {yScale(global_state.tokens.length - found.y)}px;
                    left: {xScale(found.x)}px;
                    display: {visible
                        ? 'block'
                        : 'none'
                    };
                    width: {box_width}px;
                    height: {box_height}px;
                    border: 1px solid #665191;
                "
            ></div>
            <div
                class="tooltip"
                style="
                    top: {yScale(global_state.tokens.length - found.y) + box_height}px;
                    left: {xScale(found.x) + box_width}px;
                    display: {visible
                        ? 'block'
                        : 'none'
                    };
                "
            >
                <h1 class="tooltip-heading">Score: {(found.score).toFixed(3)}</h1>
                source: {found.source}<br />
                destination: {found.destination}<br />
            </div>
        {/if}

        <svg {width} {height}>
            {#each data as element, i}
                <rect
                    role="grid"
                    tabindex="0"
                    x={xScale(element.x)}
                    y={yScale(global_state.tokens.length - element.y)}
                    width={box_width}
                    height={box_height}
                    fill={`${RGB(element.score)}`}
                    stroke={`${RGB(element.score)}`}
                    onmousemove={() => {
                        visible=true;
                        found = data[i];
                    }}
                    onblur={() => {
                        visible=false;
                        found=null;
                    }}
                    onmouseout={() => {
                        visible=false;
                        found=null;
                    }}
                />
            {/each}

            <!-- Y label -->
            <text
                x={xScale(0)}
                y={yScale(global_state.tokens.length / 2)}
                text-anchor="middle"
                transform="rotate(-90, {xScale(0)- 10}, {yScale(global_state.tokens.length / 2)})"
            >{ylabel}</text>

            <!-- X label -->
            <text
                x={xScale(global_state.tokens.length / 2)}
                y={yScale(0)}
                dy=15
                text-anchor="middle"
                transform="rotate(0, {xScale(0)}, {yScale(global_state.tokens.length / 2)})"
            >{xlabel}</text>
        </svg>
    </div>
</div>

<style>
    .chart {
        height: 100%;
        position: relative;
    }

    .rect {
        position: absolute;
        pointer-events: none;
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
        text-align: left;
    }

    .tooltip-heading {
        color: black;
    }
</style>