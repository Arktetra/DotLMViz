<script>
	let { state = $bindable(), leftlabel = '', rightlabel = '', style = '' } = $props();
</script>

<div
	class="flex flex-row items-center justify-evenly space-x-2 text-sm font-bold text-theme {style}"
>
	{#if leftlabel}
		<label for={leftlabel}>{leftlabel}</label>
	{/if}
	<input
		type="range"
		class="duration-400 inline-block w-7 cursor-pointer rounded-md accent-theme transition-all"
		min={0}
		max={1}
		step={1}
		onclick={() => (state = !state)}
		value={state ? 0 : 1}
	/>
	{#if rightlabel}
		<label for={rightlabel}>{rightlabel}</label>
	{/if}
</div>
