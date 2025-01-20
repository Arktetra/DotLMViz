<script lang="ts">
    let {x, y, r, fill} = $props();
</script>

<circle
    cx={x}
    cy={y}
    r={r}
    fill={fill}
/>

<style>
    circle {
        opacity: 0.8;
    }
</style>