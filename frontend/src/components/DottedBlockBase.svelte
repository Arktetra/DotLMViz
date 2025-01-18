<script lang="ts">
    const { label = "Untitled", borderSize = "1.5px", inStyle = "", titStyle = "", children = null } = $props()
</script>

<div
    style="border-width: {borderSize};"
    class="m-2 relative rounded-xl border-dashed border-theme transition-colors duration-200"
>
    <span class="absolute top-[-1.8rem] left-[50%] translate-x-[-50%] text-gray-500 uppercase {titStyle}">{label}</span>
    
    <div class="flex items-center {inStyle}">
        {#if children}
            {@render children()}
        {/if}
    </div>

</div>
