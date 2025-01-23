<script lang="ts">
    import { scaleLinear } from "d3-scale";
    import Circle from "./Circle.svelte";
	import { onMount } from "svelte";

    let { data } = $props<{data: Array<Array<number>>}>(); // The data needs to be accessed by indexing as data[0]. Why?

    function findMinMax(array: Array<Array<number>>, dim: number) {
        $inspect(data);
        let elements = array.map((element: Array<number>) => {
            return element[dim];
        });

        console.log([Math.min.apply(null, elements), Math.max.apply(null, elements)]);

        return [Math.min.apply(null, elements), Math.max.apply(null, elements)];
    }

    let width = $state(500);
    let height = $state(266);
    const padding = { top: 20, right: 15, bottom: 20, left: 20};

    let x_minmax = findMinMax(data[0], 0);
    let y_minmax = findMinMax(data[0], 1);
    let x_extra_space = (x_minmax[1] - x_minmax[0]) / 10;   // for some extra space around the data points
    let y_extra_space = (y_minmax[1] - y_minmax[0]) / 10;

    let xScale = $derived(scaleLinear()
        .domain([x_minmax[0] - x_extra_space, x_minmax[1] + x_extra_space])
        .range([padding.left, width - padding.right])
    );

    let yScale = $derived(scaleLinear()
        .domain([y_minmax[0] - y_extra_space, y_minmax[1] + y_extra_space])
        .range([height - padding.bottom, padding.top])
    );

    onMount(() => {
        const chart = document.querySelector(".chart");

        if (chart) {
            height = chart.clientHeight;
        }
    });

    // $inspect(data);
</script>

<div class="chart" bind:clientWidth={width}>
    <svg {width} {height}>
        <g class="bars">
            {#each data[0] as point}
                <Circle
                    x={xScale(point[0])}
                    y={yScale(point[1])}
                    r={5}
                    fill={"#6a6ef0"}
                />
            {/each}
        </g>
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