<script lang="ts">
	let { x, y, r, fill, stroke = 'blue', strokeWidth = 1 } = $props();
</script>

<circle cx={x} cy={y} {r} {fill} {stroke} stroke-width={strokeWidth} />

<style>
	circle {
		opacity: 0.8;
	}
</style>
