<script lang="ts">
	let { count, stackAxis = 'Horizontal', style = "", activeIndex = $bindable(), clickEventCb = null } = $props();
</script>

<div
	class="mb-6 flex items-center justify-evenly font-main-a rounded-md border border-theme bg-theme-w
    {stackAxis == 'Horizontal' ? 'flex-row' : 'flex-col'} {style}"
>
	{#each { length: count } as i, ind}
		<button
			onclick={() => {
				activeIndex = ind;
				if (clickEventCb != null) {
					clickEventCb();
				}
			}}
			class="h-full text-md rounded-sm p-1 px-3 font-bold {activeIndex == ind
				? ' bg-theme text-theme-w '
				: ' bg-none text-theme '}"
		>
			{ind}
		</button>
	{/each}
</div>
