<script lang="ts">
    let {x, y, r, fill, stroke="blue", strokeWidth=1} = $props();
</script>

<circle
    cx={x}
    cy={y}
    r={r}
    fill={fill}
    stroke={stroke}
    stroke-width={strokeWidth}
/>

<style>
    circle {
        opacity: 0.8;
    }
</style>