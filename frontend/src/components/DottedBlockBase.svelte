<script lang="ts">
	const {
		label = 'Untitled',
		borderSize = '1.5px',
		inStyle = '',
		titStyle = '',
		children = null
	} = $props();
</script>

<div
	style="border-width: {borderSize};"
	class="relative m-2 rounded-xl border-dashed border-theme transition-colors duration-200"
>
	<span
		class="absolute left-[50%] top-[-1.8rem] translate-x-[-50%] uppercase text-gray-500 {titStyle}"
		>{label}</span
	>

	<div class="flex items-center {inStyle}">
		{#if children}
			{@render children()}
		{/if}
	</div>
</div>
