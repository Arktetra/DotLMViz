<script lang="ts">
	import '../app.css';
	import Navbar from '../modules/Navbar.svelte';

	let { children } = $props();
</script>

<Navbar />
{@render children()}
