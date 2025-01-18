<script lang="ts">
	let { count, stackAxis = 'Horizontal', activeIndex = $bindable() } = $props();
</script>

<div
	class="mb-10 flex items-center justify-evenly rounded-md border border-theme bg-theme-w
    {stackAxis == 'Horizontal' ? 'flex-row' : 'flex-col'}"
>
	{#each { length: count } as i, ind}
		<button
			onclick={() => {
				activeIndex = ind;
			}}
			class="text-md rounded-sm p-1 px-3 font-bold {activeIndex == ind
				? ' bg-theme text-theme-w '
				: ' bg-none text-theme '}"
		>
			{ind}
		</button>
	{/each}
</div>
