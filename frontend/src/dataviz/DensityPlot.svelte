<script lang="ts">
	import { onMount } from 'svelte';
	import { data } from '../state.svelte';
	import { curveBasis, extent, line, scaleLinear, scaleOrdinal } from 'd3';
	import Axis from './Axis.svelte';
	import CategoryLegend from './CategoryLegend.svelte';

	let {
		pre,
		post
	}: {
		pre: [number, number][];
		post: [number, number][];
	} = $props();

	let width = $state(500),
		height = $state(266),
		margin = { left: 25, right: 20, top: 20, bottom: 20 },
		legendData = ['before', 'after'];

	let xPreMinMax = $derived(extent(pre, (d) => d[0]) as [number, number]),
		yPreMinMax = $derived(extent(pre, (d) => d[1]) as [number, number]),
		xPostMinMax = $derived(extent(post, (d) => d[0]) as [number, number]),
		yPostMinMax = $derived(extent(post, (d) => d[1]) as [number, number]);

	let xMin = $derived(Math.min(xPreMinMax[0], xPostMinMax[0])),
		xMax = $derived(Math.max(xPreMinMax[1], xPostMinMax[1])),
		yMin = $derived(Math.min(yPreMinMax[0], yPostMinMax[0])),
		yMax = $derived(Math.max(yPreMinMax[1], yPostMinMax[1]));

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

	let colorScale = $derived(scaleOrdinal().domain(legendData).range(['#ffaaaa', '#aaaaff']));

	let lineGenerator = $derived(
		xScale && yScale
			? line()
					.curve(curveBasis)
					.x((d) => xScale(d[0]))
					.y((d) => yScale(d[1]))
			: null
	);

	onMount(() => {
		const chart = document.querySelector('.chart');

		if (chart) {
			height = chart.clientHeight;
		}
	});
</script>

<div class="chart" bind:clientWidth={width}>
	{#if pre && post && xScale && yScale && lineGenerator}
		<svg {width} {height}>
			<Axis {xScale} {yScale} {margin} {width} {height} />

			<path
				d={lineGenerator(pre)}
				fill="#ffaaaa"
				opacity="0.5"
				stroke="#ff0000"
				stroke-width="1"
				stroke-linejoin="round"
			/>
			<path
				d={lineGenerator(post)}
				fill="#aaaaff"
				opacity="0.5"
				stroke="#0000ff"
				stroke-width="1"
				stroke-linejoin="round"
			/>

			<g transform="translate({width - margin.right - 50}, {0})">
				<CategoryLegend {legendData} legendColorFunction={colorScale} />
			</g>
		</svg>
	{/if}
</div>

<style>
	.chart {
		height: 100%;
	}

	svg {
		background-color: aliceblue;
		color: #6a6af0;
	}
</style>
