<script>
    let { state = $bindable(), leftlabel = "", rightlabel = "", style="" } = $props()
</script>

<div class="flex flex-row justify-evenly items-center text-theme text-sm font-bold {style}">
    {#if leftlabel}
        <label>{leftlabel}</label>
    {/if}
    <input type="range" class="mx-2 w-7 inline-block cursor-pointer transition-all duration-200 accent-theme rounded-md" min={0} max={1} step={1} onchange={() => state = !state} />
    {#if rightlabel}
        <label>{rightlabel}</label>
    {/if}
</div>