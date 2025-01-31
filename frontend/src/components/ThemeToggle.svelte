<script>
    let { state = $bindable(), leftlabel = "", rightlabel = "", style="" } = $props()
</script>

<div class="flex flex-row justify-evenly items-center text-theme text-sm font-bold space-x-2 {style}">
    {#if leftlabel}
        <label for={leftlabel}>{leftlabel}</label>
    {/if}
    <input type="range" class="w-7 inline-block cursor-pointer transition-all duration-200 accent-theme rounded-md" min={0} max={1} step={1} onchange={() => state = !state} />
    {#if rightlabel}
        <label for={rightlabel}>{rightlabel}</label>
    {/if}
</div>