<script lang="ts">
	import { extent } from 'd3';
	import { scaleLinear } from 'd3-scale';
	import { onMount } from 'svelte';

	let {
		data,
		vmin = '#ff7777',
		vmax = '#7777ff'
	}: {
		data: NeuronOutputData;
		vmin?: string;
		vmax?: string;
	} = $props();

	let width = $state(300);

	let found: NeuronOutput | null = $state(null);
	let visible: boolean = $state(false);
	let hoveredIndex: number | null = $state(null);
	let position: { x: number; y: number } | null = $state(null);

	let minmax = $derived(extent(data, (d) => d.score) as unknown as [number, number]);
	// let min = minmax[0];
	// let max = minmax[1];

	let redScale = $derived(scaleLinear<string>().domain([minmax[0], 0]).range([vmin, 'white']));

	let blueScale = $derived(scaleLinear<string>().domain([0, minmax[1]]).range(['white', vmax]));

	function colorScale(val: number) {
		if (val < 0) {
			return redScale(val);
		} else {
			return blueScale(val);
		}
	}

	$effect(() => {
		$inspect(position);
	});
</script>

<div class="chart" bind:clientWidth={width}>
	<div class="relative">
		{#if found}
			<!-- {console.log("changed")} -->
			<div
				class="tooltip"
				style="
                    top: {position?.y}px;
                    left: {position?.x}px;
                    display: {visible ? 'block' : 'none'};
                "
			>
				<h1 class="tooltip-heading">{found.score.toFixed(3)}</h1>
			</div>
		{/if}

		<div class="neuron">
			{#each data as datum, i}
				<div
					aria-hidden={true}
					class="token"
					style="
                        background-color: {`${colorScale(datum.score)}`};
                        opacity: {hoveredIndex === i ? 0.8 : 1.0};
                    "
					onmousemove={(evt) => {
						let chart = document.querySelector('.chart')?.getBoundingClientRect();
						position = { x: evt.clientX - chart!.left, y: evt.clientY - chart!.top };
						found = datum;
						visible = true;
						hoveredIndex = i;
					}}
					onblur={() => {
						visible = false;
						found = null;
					}}
					onmouseout={() => {
						visible = false;
						found = null;
					}}
				>
					{datum.token}
				</div>
			{/each}
		</div>
	</div>
</div>

<style>
	.chart {
		height: 100%;
		position: relative;
	}

	.neuron {
		position: absolute;
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

	.neuron {
		background-color: #eeeeff;
		display: flex;
		flex-wrap: wrap;
	}

	.token {
		white-space: pre;
		background-color: #eeffee;
		border: 1px solid #eeeeff;
	}
</style>
