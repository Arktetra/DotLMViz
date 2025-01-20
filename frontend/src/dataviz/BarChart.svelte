<script lang="ts">
	import { scaleLinear } from 'd3-scale';

	let { tokens } = $props();

	let width = $state(500);
	let height = 266;

	const padding = { top: 20, right: 15, bottom: 20, left: 25 };

	// probabilities along x-axis
	let xScale = $derived(
		scaleLinear()
			.domain([0, 1])
			.range([padding.left, width - padding.right])
	);

	// tokens along y-axis
	let yScale = $derived(
		scaleLinear()
			.domain([0, tokens.length])
			.range([height - padding.bottom, padding.top])
	);

	let innerWidth = $derived(width - (padding.left + padding.right));
	let barWidth = $derived(innerWidth / tokens.length);
</script>

<div class="chart" bind:clientWidth={width}>
	<svg {width} {height}>
		<g class="bars"> </g>
	</svg>
</div>
