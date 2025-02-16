<script>
	const { href, ostyle = '' } = $props();
	import { QuestionCircleSolid } from 'flowbite-svelte-icons';
</script>

<a
	{href}
	title={href}
	target="_blank"
	on:click|stopPropagation
	class="absolute end-1 top-1 text-theme-w hover:scale-105 {ostyle}"
>
	<QuestionCircleSolid size={'sm'} />
</a>
