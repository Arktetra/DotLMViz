<script lang="ts">
	const {
		label = null,
		type = 'text',
		maxlen = 100,
		placeholder = 'Enter text here..',
		value = $bindable(),
		inpEventCb = null
	} = $props();

	const inpChange = (v: any) => {
		if (inpEventCb) inpEventCb(v);
	};
</script>

{#if label}
	<label for={label} class="text-theme">{label}:</label>
{/if}
<input
	{type}
	{value}
	{placeholder}
	maxlength={maxlen}
	oninput={(e: any) => inpChange(e.target.value)}
	class="m-2 min-w-[20rem] xl:min-w-[35rem] rounded-md border border-theme-g-alt focus:border-theme p-[0.25rem] text-lg text-theme outline-none"
/>
