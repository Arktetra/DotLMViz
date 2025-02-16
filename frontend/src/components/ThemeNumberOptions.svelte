<script lang="ts">
	let {
		count,
		stackAxis = 'Horizontal',
		style = '',
		activeIndex = $bindable(),
		clickEventCb = null
	} = $props();
</script>

<div
	class="mb-6 flex items-center justify-between rounded-md border border-theme bg-theme-w 
    {stackAxis == 'Horizontal' ? 'flex-row' : 'flex-col'} {style}"
>
	{#each { length: count } as i, ind}
		<button
			onclick={() => {
				activeIndex = ind;
				clickEventCb && clickEventCb()
			}}
			class="text-md rounded-sm p-1 px-3 font-medium {activeIndex == ind
				? ' bg-theme text-theme-w hover:bg-theme-alt'
				: ' bg-none text-theme hover:bg-theme-g-alt'}"
		>
			{ind}
		</button>
	{/each}
</div>
