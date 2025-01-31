<script lang="ts">
	import type { ScaleLinear } from 'd3-scale';

	let {
		xScale,
		yScale,
		margin,
		height,
		width,
		ticksNumber = 5
	}: {
		xScale: ScaleLinear<number, number>;
		yScale: ScaleLinear<number, number>;
		margin: { left: number; right: number; top: number; bottom: number };
		height: number;
		width: number;
		ticksNumber?: number;
	} = $props();
</script>

<g transform="translate(0, {height - margin.bottom})">
	<line stroke="black" x1={margin.left} x2={width - margin.right} />

	{#each xScale.ticks(ticksNumber) as tick}
		<line stroke="black" x1={xScale(tick)} x2={xScale(tick)} y1={0} y2={6} />
	{/each}

	{#each xScale.ticks(ticksNumber) as tick}
		<text font-size="12px" fill="black" text-anchor="middle" x={xScale(tick)} y={16}>
			{tick}
		</text>
	{/each}
</g>

<g transform="translate({margin.left}, 0)">
	<line stroke="black" y1={yScale(yScale.domain()[0])} y2={yScale(yScale.domain()[1])} />

	{#each yScale.ticks(ticksNumber) as tick}
		{#if tick !== 0}
			<line stroke="black" x1={0} x2={-6} y1={yScale(tick)} y2={yScale(tick)} />
		{/if}

		<text
			fill="black"
			text-anchor="end"
			font-size="10"
			dominant-baseline="middle"
			x={-9}
			y={yScale(tick)}
		>
			{tick}
		</text>
	{/each}
</g>
