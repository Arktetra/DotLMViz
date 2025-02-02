<script lang="ts">
	let { legendData, legendColorFunction } = $props();
</script>

{#each legendData as d, i}
	<rect x="20" y={i * 20 + 10} width="10" height="10" fill={legendColorFunction(d)} />
	<text x="35" y={18 + i * 20}>
		{d[0] + d.slice(1).toLowerCase()}
	</text>
{/each}
