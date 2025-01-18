<script lang="ts">
    let { count, stackAxis = "Horizontal", activeIndex = $bindable() } = $props()
</script>

<div 
    class="mb-10 flex justify-evenly items-center border border-theme bg-theme-w rounded-md 
    {stackAxis == 'Horizontal' ? "flex-row" : "flex-col"}"
>
    {#each {length : count} as i, ind}
        <button 
            onclick={() => {activeIndex = ind}}
            class="p-1 px-3 rounded-sm text-md font-bold {activeIndex == ind ? ' text-theme-w bg-theme ' : ' text-theme bg-none '}"
        >
            {ind}
        </button>
    {/each}
</div>