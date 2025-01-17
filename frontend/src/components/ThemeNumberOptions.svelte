<script lang="ts">
    let { count, stackAxis = "Horizontal", activeIndex = $bindable() } = $props()
</script>

<div 
    class="mb-20 p-1 flex justify-evenly items-center border border-theme bg-theme-w rounded-md 
    {stackAxis == 'Horizontal' ? "flex-row" : "flex-col"}"
>
    {#each {length : count} as i, ind}
        <button 
            onclick={() => { console.log("Cliked" + ind);   activeIndex = ind}}
            class="p-1 px-3 rounded-sm text-md {activeIndex == ind ? ' font-bold text-theme-w bg-theme ' : ' font-light text-theme bg-none '}"
        >
            {ind}
        </button>
    {/each}
</div>